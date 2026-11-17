"""Export the trained model so the web page can run it in the browser.

GitHub Pages serves static files only, so there is no Python server behind the
public site. This writes docs/model.json (every decision tree as flat arrays)
and docs/meta.json (lookups and reference tables); docs/predictor.js does the
rest: features, prediction and SHAP explanations.
"""
import json
import shutil
from pathlib import Path

import lightgbm as lgb

from features import FEATURES, GROUPS, GROUP_ORDER

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def flatten(tree):
    """Nested LightGBM tree -> flat arrays. A child >= 0 is an internal node,
    a child < 0 is leaf number ~child."""
    n_leaves = tree["num_leaves"]
    n_int = n_leaves - 1
    f, t, c = [0] * n_int, [None] * n_int, [0] * n_int
    left, right, n = [0] * n_int, [0] * n_int, [0] * n_int
    v, m = [0.0] * n_leaves, [0] * n_leaves

    def ref(node):
        return node["split_index"] if "split_index" in node else ~node.get("leaf_index", 0)

    def walk(node):
        if "split_index" in node:
            i = node["split_index"]
            f[i] = node["split_feature"]
            if node["decision_type"] == "==":           # categorical: left if in set
                c[i] = 1
                t[i] = [int(x) for x in str(node["threshold"]).split("||")]
            else:                                       # numeric: left if value <= t
                t[i] = node["threshold"]
            n[i] = node["internal_count"]
            left[i], right[i] = ref(node["left_child"]), ref(node["right_child"])
            walk(node["left_child"])
            walk(node["right_child"])
        else:
            i = node.get("leaf_index", 0)
            v[i] = node["leaf_value"]
            m[i] = node.get("leaf_count", 0)

    walk(tree["tree_structure"])
    return {"f": f, "t": t, "c": c, "l": left, "r": right, "n": n, "v": v, "m": m}


def main():
    booster = lgb.Booster(model_file=str(ROOT / "models" / "model.txt"))
    dump = booster.dump_model()
    assert dump["feature_names"] == FEATURES
    model = {
        "features": FEATURES,
        "groups": [GROUPS[f] for f in FEATURES],
        "group_order": GROUP_ORDER,
        "trees": [flatten(t) for t in dump["tree_info"]],
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    shutil.copyfile(ROOT / "models" / "meta.json", DOCS / "meta.json")
    print(f"{len(model['trees'])} trees -> {DOCS / 'model.json'} "
          f"({(DOCS / 'model.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
