/* DelayRadar predictor: runs the trained LightGBM model in the browser.
 *
 * model.json holds every decision tree as flat arrays (see src/export_web.py);
 * meta.json holds the lookups and reference tables written by src/train.py.
 * This file mirrors src/features.py (inputs) and src/app.py (explanations),
 * so the static site gives the same answers as the Python API.
 */
const Predictor = (() => {
  let META, MODEL;
  const DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'];
  const DAY = 864e5;
  const f32 = Math.fround;                       // training used 32-bit floats
  const pct = x => (x * 100).toFixed(0) + '%';
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  async function load(base = '') {
    [META, MODEL] = await Promise.all(
      ['meta.json', 'model.json'].map(f => fetch(base + f).then(r => r.json())));
    MODEL.catIndex = {};
    for (const c of ['airline', 'origin', 'dest'])
      MODEL.catIndex[c] = new Map(META.categories[c].map((v, i) => [v, i]));
    for (const t of MODEL.trees)                 // categorical thresholds -> sets
      t.t = t.t.map((th, i) => t.c[i] ? new Set(th) : th);
    return META;
  }

  /* ---------- features (mirrors src/features.py) ---------- */
  const utc = (y, m, d) => Date.UTC(y, m - 1, d);
  function nthWeekday(year, month, weekday, n) {  // weekday: Monday = 0; n = -1 for last
    const dow = t => (new Date(t).getUTCDay() + 6) % 7;
    if (n > 0) {
      const first = utc(year, month, 1);
      return first + (((weekday - dow(first)) % 7 + 7) % 7 + 7 * (n - 1)) * DAY;
    }
    const last = utc(month === 12 ? year + 1 : year, month % 12 + 1, 1) - DAY;
    return last - ((dow(last) - weekday + 7) % 7) * DAY;
  }
  const holidays = y => [utc(y, 1, 1), nthWeekday(y, 5, 0, -1), utc(y, 7, 4),
    nthWeekday(y, 9, 0, 1), nthWeekday(y, 11, 3, 4), utc(y, 12, 25)];
  function holidayGap(t) {
    const y = new Date(t).getUTCFullYear();
    let best = 14;
    for (const yy of [y - 1, y, y + 1])
      for (const h of holidays(yy)) best = Math.min(best, Math.abs(Math.round((t - h) / DAY)));
    return best;
  }

  function featureRow(q, minutes) {
    const route = META.routes[q.origin + '-' + q.dest];
    const arr = (minutes + route.shift) % 1440;
    const [y, m, d] = q.date.split('-').map(Number);
    const t = utc(y, m, d);
    const depH = Math.floor(minutes / 60), arrH = Math.floor(arr / 60);
    const cat = (name, v) => MODEL.catIndex[name].has(v) ? MODEL.catIndex[name].get(v) : NaN;
    const row = {
      airline: cat('airline', q.airline), origin: cat('origin', q.origin), dest: cat('dest', q.dest),
      month: m, day_of_week: (new Date(t).getUTCDay() + 6) % 7 + 1,
      dep_hour: f32(depH + (minutes % 60) / 60), arr_hour: f32(arrH + (arr % 60) / 60),
      distance: f32(route.distance), holiday_gap: holidayGap(t),
      route_daily: f32(META.route_daily[q.origin + '-' + q.dest] || 0),
      origin_hour_deps: f32((META.origin_hour_deps[q.origin] || [])[depH] || 0),
      dest_hour_arrs: f32((META.dest_hour_arrs[q.dest] || [])[arrH] || 0),
    };
    return { x: MODEL.features.map(f => row[f]), row, arr, dow: row.day_of_week, month: m };
  }

  /* ---------- prediction ---------- */
  const goesLeft = (tree, i, x) => {
    const v = x[tree.f[i]];
    return tree.c[i] ? tree.t[i].has(v) : v <= tree.t[i];
  };
  function rawScore(x) {
    let s = 0;
    for (const t of MODEL.trees) {
      let i = t.f.length ? 0 : -1;
      while (i >= 0) i = goesLeft(t, i, x) ? t.l[i] : t.r[i];
      s += t.v[~i];
    }
    return s;
  }

  /* ---------- SHAP values (TreeSHAP, Lundberg et al., as implemented in LightGBM) ---------- */
  function extend(path, depth, zero, one, feat) {
    path[depth] = { f: feat, z: zero, o: one, w: depth === 0 ? 1 : 0 };
    for (let i = depth - 1; i >= 0; i--) {
      path[i + 1].w += one * path[i].w * (i + 1) / (depth + 1);
      path[i].w = zero * path[i].w * (depth - i) / (depth + 1);
    }
  }
  function unwind(path, depth, idx) {
    const one = path[idx].o, zero = path[idx].z;
    let next = path[depth].w;
    for (let i = depth - 1; i >= 0; i--) {
      if (one !== 0) {
        const tmp = path[i].w;
        path[i].w = next * (depth + 1) / ((i + 1) * one);
        next = tmp - path[i].w * zero * (depth - i) / (depth + 1);
      } else {
        path[i].w = path[i].w * (depth + 1) / (zero * (depth - i));
      }
    }
    for (let i = idx; i < depth; i++) { path[i].f = path[i + 1].f; path[i].z = path[i + 1].z; path[i].o = path[i + 1].o; }
  }
  function unwoundSum(path, depth, idx) {
    const one = path[idx].o, zero = path[idx].z;
    let next = path[depth].w, total = 0;
    for (let i = depth - 1; i >= 0; i--) {
      if (one !== 0) {
        const tmp = next * (depth + 1) / ((i + 1) * one);
        total += tmp;
        next = path[i].w - tmp * zero * ((depth - i) / (depth + 1));
      } else {
        total += (path[i].w / zero) / ((depth - i) / (depth + 1));
      }
    }
    return total;
  }
  function treeShap(t, x, phi, node, depth, parent, pZero, pOne, pFeat) {
    const path = parent.slice(0, depth).map(e => ({ ...e }));
    extend(path, depth, pZero, pOne, pFeat);
    if (node < 0) {
      const leaf = t.v[~node];
      for (let i = 1; i <= depth; i++)
        phi[path[i].f] += unwoundSum(path, depth, i) * (path[i].o - path[i].z) * leaf;
      return;
    }
    const left = goesLeft(t, node, x);
    const hot = left ? t.l[node] : t.r[node], cold = left ? t.r[node] : t.l[node];
    const count = n => n < 0 ? t.m[~n] : t.n[n];
    const w = t.n[node];
    let inZero = 1, inOne = 1, idx = 0;
    for (; idx <= depth; idx++) if (path[idx].f === t.f[node]) break;
    if (idx !== depth + 1) {
      inZero = path[idx].z; inOne = path[idx].o;
      unwind(path, depth, idx);
      depth -= 1;
    }
    treeShap(t, x, phi, hot, depth + 1, path, count(hot) / w * inZero, inOne, t.f[node]);
    treeShap(t, x, phi, cold, depth + 1, path, count(cold) / w * inZero, 0, t.f[node]);
  }
  function shap(x) {
    const phi = new Array(MODEL.features.length).fill(0);
    for (const t of MODEL.trees) if (t.f.length) treeShap(t, x, phi, 0, 0, [], 1, 1, -1);
    return phi;
  }

  /* ---------- explanation (mirrors src/app.py) ---------- */
  function explain(q, feat, phi, p, minutes) {
    const sums = {};
    MODEL.groups.forEach((g, j) => { sums[g] = (sums[g] || 0) + phi[j]; });
    const logit = Math.log(p / (1 - p));
    const al = META.airlines[q.airline], o = META.airports[q.origin] || {}, d = META.airports[q.dest] || {};
    const key = q.origin + '-' + q.dest, route = META.routes[key];
    const hour = Math.floor(minutes / 60);
    const hr = (META.rate_by_hour[hour] || {}).rate;
    const mon = (META.rate_by_month[feat.month] || {}).rate;
    const gap = feat.row.holiday_gap;
    let season = mon != null ? `${MONTHS[feat.month - 1]} flights run late ${pct(mon)} of the time.` : '';
    if (gap <= 3) season += ` This date is within ${gap} day${gap !== 1 ? 's' : ''} of a holiday travel peak.`;
    const detail = {
      'Departure time': hr != null ? `Flights leaving in the ${String(hour).padStart(2, '0')}:00 hour run late ${pct(hr)} of the time; delays build up through the day.` : '',
      'Airline': `${al.name} arrives late on ${pct(al.rate)} of its flights.`,
      'Origin airport': o.dep_rate != null ? `${pct(o.dep_rate)} of flights leaving ${q.origin} arrive late.` : '',
      'Destination airport': d.arr_rate != null ? `${pct(d.arr_rate)} of flights into ${q.dest} arrive late.` : '',
      'Season': season,
      'Day of week': `${DOW[feat.dow - 1]} flights run late ${pct(META.rate_by_dow[feat.dow].rate)} of the time.`,
      'Route': `${route.distance.toLocaleString('en-US')} miles, about ${Math.max(1, Math.round(META.route_daily[key] || 0))} flights a day; ${pct(route.rate)} of them arrive late.`,
      'Airport congestion': `${q.origin} schedules about ${feat.row.origin_hour_deps.toFixed(0)} departures that hour; ${q.dest} about ${feat.row.dest_hour_arrs.toFixed(0)} arrivals when you land.`,
    };
    return MODEL.group_order.map(name => {
      const without = sigmoid(logit - (sums[name] || 0));
      return { factor: name, points: Math.round((p - without) * 1000) / 10, detail: detail[name] };
    }).sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
  }

  /* ---------- public: same shape as the Python API's /api/predict ---------- */
  function predict(q) {
    const route = META.routes[q.origin + '-' + q.dest];
    if (!route) return { error: `No ${q.origin} to ${q.dest} flights in the training data.` };
    if (!META.airlines[q.airline]) return { error: `Unknown airline ${q.airline}.` };
    const [hh, mm] = q.time.split(':').map(Number);
    const minutes = hh * 60 + mm;
    const feat = featureRow(q, minutes);
    const p = sigmoid(rawScore(feat.x));
    const band = (META.bands.find(b => p < b.hi) || META.bands[META.bands.length - 1]).band;
    let rank = META.percentiles.findIndex(v => v >= p);          // bisect_left
    if (rank < 0) rank = META.percentiles.length;
    const hourly = [];
    for (let h = 5; h < 24; h++)
      hourly.push({ hour: h, risk: sigmoid(rawScore(featureRow(q, h * 60 + mm).x)) });
    return {
      risk: p, band, base_rate: META.base_rate,
      percentile: Math.max(0, Math.min(99, rank - 1)),
      factors: explain(q, feat, shap(feat.x), p, minutes),
      hourly,
      arrival: String(Math.floor(feat.arr / 60)).padStart(2, '0') + ':' + String(feat.arr % 60).padStart(2, '0'),
      distance: route.distance, route_rate: route.rate,
      airline_flies_route: route.airlines.includes(q.airline),
      airline_name: META.airlines[q.airline].name,
    };
  }

  return { load, predict };
})();
