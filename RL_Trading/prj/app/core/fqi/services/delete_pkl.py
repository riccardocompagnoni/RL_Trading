import os
if __name__ == '__main__':

    from pathlib import Path;
    import json, re

    ROOT = Path("/home/a2a/a2a/RL_Trading/results")
    for d in ROOT.iterdir():
        if not (d.is_dir() and d.name.startswith(("delta", "xgb"))): continue
        p = d / "parameters_opt.json"
        try:
            it = int(str(json.load(p.open())["iterations"]).strip())
        except Exception:
            continue
        keep = f"Policy_iter{it}.pkl"
        print(it)
        print(keep)
        for f in d.rglob("Policy_iter*.pkl"):
            print(f)
            if re.fullmatch(r"Policy_iter\d+\.pkl", f.name) and f.name != keep:
                f.unlink()