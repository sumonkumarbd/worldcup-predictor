# World Cup 2026 Match Predictor — Report

## What this is
A from-scratch system that predicts the **full goal-by-goal scoreline probability**
for any international football fixture, plus — for knockout matches — **who advances**
after extra time and penalties. Built and validated in phases; this report documents
each phase honestly, including the ones that didn't work.

## Data
[`martj42/international_results`](https://github.com/martj42/international_results)
(CC0 / public domain) — 49,000+ men's full-international matches from 1872 to today,
plus its companion `shootouts.csv` (678 penalty-shootout records). Both are
community-maintained and auto-updated daily, which means re-running the exact same
analysis on different days can give very slightly different numbers (we hit this —
see the Phase 2c note below). This isn't a bug, just a property of building on a
live-updating dataset mid-tournament.

## Architecture (current, final state)
1. **Elo ratings** computed match-by-match across the full 154-year history —
   K-factor scaled by tournament importance, goal-difference multiplier, +100 Elo
   home-advantage adjustment for non-neutral venues.
2. **Rolling form** — each team's average goals scored/conceded over its last 10
   matches as of just before the match (no lookahead).
3. **Head-to-head record** — win rate, draw rate, and average goal difference from
   each pair's last 5 meetings.
4. **Gradient-boosted Poisson regression** (`HistGradientBoostingRegressor(loss="poisson")`)
   predicting each team's expected goals (λ) from the above features.
5. **Dixon-Coles correlation correction** (ρ) applied to the independent-Poisson
   scoreline matrix, fixing the known under-prediction of low scores (0-0, 1-1).
6. **Knockout/penalty-shootout layer** — converts the 90-minute scoreline
   distribution into "who advances," for matches that can't end in a draw.
7. **SHAP-based explanations** — per-prediction, human-readable reasons.

---

## Phase 0 — Original baseline (Elo + Poisson GBM only, no Dixon-Coles, no H2H)
Trained on 2000-2023 (22,799 matches), tested on 2024-2026 (2,610 genuinely
held-out matches).

| | Log-loss |
|---|---|
| GBM + Poisson model | 0.8702 |
| Elo-only baseline (logistic regression on Elo diff) | 0.8669 |
| No-information floor (historical base rates) | 1.0537 |

Calibration error: 0.0216. Bootstrap 95% CI for (model − baseline) log-loss:
**[-0.0031, 0.0098]** — spans zero, i.e. not statistically distinguishable from a
plain Elo-only baseline at calling win/draw/loss. The model's value at this stage
was being well-calibrated and producing a full scoreline distribution (which the
baseline can't do at all) — not being more *accurate* at W/D/L.

Classification accuracy (computed later, same test set): model 60.2% vs Elo-only
60.4% (tied), vs always-predict-home-win 47.5%. Top-2 accuracy 84.3%. Exact-scoreline
accuracy 12.6% (normal for football — even strong professional models sit around
12-18%). **Notable finding:** the model never once predicted "draw" as its single
most-likely outcome across all 2,610 test matches, despite draws occurring in ~24%
of them — a structural property of Poisson-style models (argmax rarely lands on the
draw class even when its probability is meaningfully high), not a bug. The
percentages should be trusted; the single "most likely" label should not be read
as "draw is impossible."

## Phase 1 — Dixon-Coles correction (isolated effect)
Fitted ρ via maximum likelihood on the training set: **ρ ≈ -0.047**, in the expected
literature direction (negative) though smaller in magnitude than the original
Dixon & Coles (1997) estimate (~-0.13) — plausible, since recent-form features here
already explain part of that low-score correlation.

Isolated effect (Dixon-Coles only, no head-to-head): log-loss 0.8699 → 0.8686
(**-0.0013**). Low-score calibration impact specifically:

| Score | n (test) | P(exact) no-ρ | P(exact) with ρ | P(draw) no-ρ | P(draw) with ρ |
|---|---|---|---|---|---|
| 0-0 | 236 | 0.0849 | 0.0899 (+0.0050) | 0.2360 | 0.2461 (+0.0101) |
| 1-1 | 267 | 0.1092 | 0.1143 (+0.0051) | 0.2373 | 0.2475 (+0.0102) |

Real, mechanistic improvement exactly where Dixon-Coles theory predicts it — kept
regardless of the overall W/D/L tie, because it fixes something specific and real.

## Phase 2a — Head-to-head features (isolated effect)
Added h2h win rate / draw rate / avg goal difference from each pair's last 5
meetings. Isolated effect (H2H only, no Dixon-Coles): log-loss 0.8699 → 0.8680
(**-0.0019**).

## Phase 2b — Home/away-split rolling form
Tried splitting recent form into home-context vs away-context separately. Caused a
regression (worse log-loss) — **reverted**, kept the simpler combined form average.

## Phase 2c — Days since last match (rest/fatigue proxy)
Isolated effect: log-loss 0.8664 → 0.8667 (**+0.0003, worse**). **Dropped** — did
not earn its place. (Re-running this same isolated comparison later under a fresh
data pull gave 0.8699 → varying combined numbers in the 0.8664-0.8667 range for what
should be the identical Phase-1+2a configuration — a ~0.0003 wobble traced to the
source dataset refreshing with new tournament results between runs, not a
methodology error. Treat any single log-loss figure in this report as ± that much
noise from day-to-day data drift during an active tournament.)

## Combined: Dixon-Coles (Phase 1) + Head-to-head (Phase 2a)
| | ll_model | ρ |
|---|---|---|
| PH0 — neither | 0.8699 | 0 |
| PH1 — ρ only | 0.8686 | -0.0462 |
| PH2a — H2H only | 0.8680 | 0 |
| Combined | 0.8664–0.8667* | ≈ -0.047 |

Effects are additive: ρ effect (-0.0013) + H2H effect (-0.0019) ≈ combined effect
(-0.0032/-0.0035); interaction term ≈ -0.00001 (negligible) — the two improvements
don't fight each other or double-count.

Bootstrap 95% CI for (combined model − Elo-only baseline) log-loss: **[-0.0069,
0.0058]** — *still spans zero*. Honest read: precision work through Phase 2a
produced a real, mechanistically-explained improvement in low-score calibration,
but has not yet produced a statistically provable edge over plain Elo at calling
the basic outcome. That remains the case after this phase.

*range reflects the data-drift wobble noted in Phase 2c.

## Phase 3 — SHAP explainability (two real bugs found and fixed)
**Bug 1 (log-space vs linear-space):** verified independently that
`shap.TreeExplainer` on this Poisson-loss model returns values in log-link space —
`exp(base_value + sum(shap_values)) == gbm.predict(X)`, not the sum directly. The
first implementation presented raw log-space SHAP values as if they were additive
goals contributions (e.g. "+0.264" implying +0.264 goals, when it actually meant a
~30% relative increase). **Fixed** by converting every SHAP value to a percentage
effect via `(exp(s) - 1) * 100` before it reaches any sentence or API field.

**Bug 2 (sign contradiction):** for closely-matched teams (e.g. DR Congo vs
Uzbekistan, raw Elo gap of just 4.3 points), the model's local SHAP attribution for
`elo_diff` came out positive on *both* sides simultaneously, even though the raw
feature value is mathematically the exact negative on each side — both teams'
explanations claimed "higher Elo than opponent." This is a real, if rare, tree-model
interaction artifact, not a feature-construction bug (verified: raw `elo_diff`
values were confirmed exact opposites; only the SHAP attribution signs disagreed).
**Fixed** by choosing direction wording from the raw feature's sign (a fact, never
ambiguous) and dropping a feature from a side's top-3 explanation whenever its SHAP
sign contradicts that raw sign — falling back to the next-ranked feature instead of
emitting a self-contradictory sentence.

**Validation:** `validate_explanation_consistency()` — an automated sweep across all
6 upcoming fixtures plus 30 random team pairs (36 cases total) — returned **0
violations**. Confirmed live (not just in a standalone test script) by checking the
running server process's start time against the source file's last-modified time.

## Phase 4 — Wired into the dashboard
`home_explanation` / `away_explanation` added as new fields to `/api/matchup` and
`/api/predictions`, surfaced as text blocks in `index.html` using only existing
design tokens. No existing field was renamed or removed.

## Phase 5 — Knockout / penalty-shootout modeling
The current 6 fixtures (see Predictions below) are Round of 32 — they cannot end in
a draw. The 90-minute scoreline model alone doesn't answer "who advances."

**Shootout data:** 678 historical shootouts (677 merged with main feature set).
Raw historical home-win rate: 54.2% (367W–310L). Logistic regression,
`shootout_home_win ~ elo_diff_pre`: coefficient 0.001647, SE 0.000575, **95% CI
[0.00052, 0.00278]**, p = 0.0042 — statistically significant, CI excludes zero, so
the Elo-based shootout model was kept (per the pre-committed rule: a
non-significant coefficient would have forced a fallback to the flat 54.2% rate).

**First-shooter check (causal sanity check):** away teams shoot first in 77.3% of
cases (524/678) — overwhelmingly more often than home teams (22.7%, 154/678),
likely a tournament-convention artifact rather than anything meaningful. Home win
rate is nearly identical regardless of who shoots first (54.6% vs 54.0%, a 0.6pp
gap) — ruling out shooting order as the driver. The ~54.2% baseline is read as a
genuine (if mild) home-side Elo skew at shootout time, not a shoots-first artifact.

**Formula:**
```
p_home_advances = p_home_90 + p_draw_90 × (p_home_ET + p_draw_ET × p_shootout_home)
p_shootout_home = sigmoid(0.1024 + 0.001647 × elo_diff_pre)
```
Extra time reuses the same scoreline-matrix machinery with λ scaled by 30/90.
Validated on all 6 current fixtures — `p_home_advances + p_away_advances` sums to
exactly 1.0 in every case (no draw possible by construction).

---

## Current predictions — Round of 32 (as of 2026-06-27)

| Match | Model H/D/A% (90 min) | Market H/D/A% | Advances (incl. ET+pens) |
|---|---|---|---|
| Panama vs England | 8.5 / 18.8 / 72.7 | 5.4 / 10.7 / 83.9 | 13.8% / 86.2% |
| Croatia vs Ghana | 67.0 / 21.3 / 11.7 | 50.5 / 29.9 / 19.6 | 82.2% / 17.8% |
| Colombia vs Portugal | 41.1 / 26.9 / 32.1 | 26.5 / 25.3 / 48.2 | 55.6% / 44.4% |
| DR Congo vs Uzbekistan | 35.4 / 29.1 / 35.4 | 60.4 / 23.0 / 16.6 | 50.4% / 49.6% |
| Jordan vs Argentina | 1.9 / 6.8 / 91.3 | 4.8 / 11.2 / 84.0 | 2.9% / 97.1% |
| Algeria vs Austria | 32.8 / 27.3 / 39.9 | 22.1 / 46.0 / 31.9 | 46.3% / 53.7% |

Per-match SHAP-based explanations are available live via `/api/matchup` and
`/api/predictions` (`home_explanation`/`away_explanation` fields) — not duplicated
here since they're generated dynamically and will drift slightly as the model
auto-refreshes.

Where model and market disagree most (Colombia/Portugal, DR Congo/Uzbekistan,
Algeria/Austria's unusually high market draw probability) — same caveat as before:
the model has no squad/lineup/injury data, so a market disagreeing strongly is
worth checking against actual team news before trusting either side blindly.

## Honest limitations — current state
- **No squad/injury/lineup data** — still the single highest-value next upgrade,
  unchanged since Phase 0.
- **International matches are infrequent** — recent-form sample sizes are small per
  team relative to club football. Treat short streaks with caution.
- **Data drift between runs** — the source dataset auto-updates during the live
  tournament; re-running the same analysis hours apart can shift log-loss by
  ~0.0003-0.0005 purely from new matches entering the training/test windows. Not a
  bug, but means small reported deltas should be read with that noise floor in mind.
- **Combined precision gain (Phase 1+2a) is not yet statistically proven** to beat
  plain Elo at win/draw/loss calling (CI still spans zero) — it is proven to fix
  low-score calibration specifically, and it's a prerequisite for the scoreline-
  distribution and knockout-advancement features, which plain Elo cannot produce
  at all.
- **Shootout model uses a single pooled coefficient** across all eras/confederations
  — hasn't been checked for whether the Elo-shootout relationship is stable across,
  say, different continents or knockout formats.

## Files in this delivery
- `predictor_core.py` — full pipeline (data, Elo, features, model, Dixon-Coles,
  knockout/shootout layer, SHAP explanations)
- `server.py` — Flask backend, auto-refresh, API
- `static/index.html` — dashboard UI
- `worldcup_predictor.py` — CLI report version
- `requirements.txt`, `UI_README.md`