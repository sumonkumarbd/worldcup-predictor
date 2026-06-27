"""
worldcup_predictor.py — CLI report (same model as the web UI in server.py).

Run:
    python3 worldcup_predictor.py

For the interactive dashboard instead, see server.py + static/index.html.
"""
import pickle
import pandas as pd

import predictor_core as pc

if __name__ == "__main__":
    state = pc.run_pipeline(force_refresh=False)
    bt = state["backtest"]

    print("=" * 60)
    print(f"BACKTEST  ({bt['n_test']} held-out matches, 2024-01-01 -> present)")
    print("=" * 60)
    print(f"Log-loss   model(GBM+Poisson)={bt['ll_model']}   Elo-only baseline={bt['ll_base']}   "
          f"base-rate floor={bt['ll_floor']}")
    print(f"Calibration error: {bt['cal_err']}")
    print(f"Bootstrap 95% CI, (model - baseline) log-loss: [{bt['ci_lo']}, {bt['ci_hi']}]  "
          f"{'(statistically real gap)' if bt['gap_significant'] else '(not statistically distinguishable from 0)'}")
    print("\nPre-committed pass/fail:")
    print(f"  (a) beats Elo-only baseline (statistically significant, CI excludes zero) : {'PASS' if bt['crit_a'] else 'FAIL'}")
    print(f"  (b) calibration error <= 0.08           : {'PASS' if bt['crit_b'] else 'FAIL'}")
    print(f"  (c) beats base-rate floor by >=3%       : {'PASS' if bt['crit_c'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if bt['overall'] else 'FAIL'}")

    print("\n" + "=" * 60)
    print("PREDICTIONS — upcoming fixtures")
    print("=" * 60)
    pred_df = pd.DataFrame(state["predictions"])
    pd.set_option("display.width", 160)
    cols = ["home", "away", "lam_home", "lam_away", "model_H", "model_D", "model_A",
            "market_H", "market_D", "market_A", "most_likely_score", "p_most_likely"]
    print(pred_df[cols].to_string(index=False))

    pred_df.drop(columns=["matrix"]).to_csv("predictions.csv", index=False)
    with open("model_state.pkl", "wb") as f:
        pickle.dump(state, f)
    print("\nSaved predictions.csv and model_state.pkl")
