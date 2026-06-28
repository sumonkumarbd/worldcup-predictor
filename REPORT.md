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

## Phase 6 — StatsBomb corners model (explored, not shipped)

**Motivation:** the goals model gives λ per team but nothing about set-piece
pressure. Corners per team per match felt like a useful extra signal (and a
natural second target from event-level data). StatsBomb's open dataset is the
only freely available event-level source for international football, so it was
the natural place to look.

**Data collected:**
StatsBomb open data — 16 senior international competitions
(`competition_international=true`, `competition_youth=false`):
WC 1958/62/70/74/86/90, WC 2018/2022, UEFA Euro 2020/2024, Copa America 2024,
AFCON 2023, Women's WC 2019/2023, UEFA Women's Euro 2022/2025.
511 matches in metadata; 504 events successfully fetched (7 HTTP 400 errors on
specific GitHub raw-content URLs — unrecoverable).

**Name mapping (men's competitions):**
78 distinct StatsBomb team names in men's competitions; **78/78 mapped**
(100%) to our `elo_final` canonical names. Two manual additions were required:

| StatsBomb | → ours | reason |
|---|---|---|
| `Congo DR` | `DR Congo` | word order swap |
| `Côte d'Ivoire` | `Ivory Coast` | accented form vs ASCII form |

All other names matched exactly or via fuzzy match (difflib cutoff 0.8).
Women's competitions (43 team names) carry no men's-data Elo/form — all got
`is_womens=1` flag and default features (`ELO_INIT=1500`, `gf/ga=GMEAN=1.4`).

**Match retention:** 504 / 504 usable (7 lost to event-fetch errors, **zero**
lost to name-mapping failures after the two manual additions).

**Feature join (men's matches):**
```
played_df exact join   307 / 332   (92.5%)  — real pre-match Elo + form
elo_final fallback       25 / 332   ( 7.5%)  — old WC matches (1958-1990)
```

**Pooled run and hypothesis tested:** the first model run pooled all 504
matches — men's and women's — with an `is_womens` binary flag and default
`ELO_INIT / GMEAN` features for the 178 women's rows. The calibration table
showed a −1.05 corner bias in the lowest-λ bucket, and the initial hypothesis
was that women's matches, whose flat default features push predicted λ
artificially low, were the cause. The men's-only re-run was run specifically
to test that hypothesis. It was wrong: removing all 178 women's rows made the
low-λ bias worse (−1.353, not better), confirming women's default features
were not the driver. The real cause — stale `elo_final` values for dissolved
nations (Soviet Union, West Germany, Czechoslovakia) concentrating in that
same bucket — was identified only after the hypothesis was refuted.

**MEN'S-ONLY model (decisive run — pre-committed go/no-go):**

Restricted to the 332 men's matches; women's rows removed from both train and
test so they contribute nothing to evaluation.

Chronological split (cutoff 2024-01-01):
```
Train: 197 matches (394 rows) — World Cup 2018/2022, Euro 2020, old WC
Test:  135 matches (270 rows) — AFCON 2023, Copa America 2024, Euro 2024
```

Model: `HistGradientBoostingRegressor(loss="poisson")`,
features: `elo_diff, team_gf, team_ga, opp_gf, opp_ga, is_home_adv`.

Baseline: flat Poisson(λ = 4.591) — train-set mean corners per team, no
features at all.

```
Mean Poisson NLL (model):    2.4919
Mean Poisson NLL (baseline): 2.5152
Difference (model − base):  −0.0233  (model better, point estimate)

Bootstrap 95% CI for (model − baseline): [−0.1396, +0.0965]
→ CI spans zero — not statistically distinguishable from baseline
```

Calibration table (10 quantile buckets of predicted λ):
```
pred λ range          pred_mean  actual_mean   n    bias
(1.123, 2.627]            1.943        3.296  27   −1.353
(2.627, 3.355]            2.951        3.481  27   −0.530
(3.355, 4.002]            3.675        4.185  27   −0.511
(4.002, 4.368]            4.188        3.741  27   +0.447
(4.368, 4.761]            4.577        4.393  28   +0.184
(4.761, 5.060]            4.874        4.346  26   +0.528
(5.060, 5.413]            5.243        5.741  27   −0.498
(5.413, 5.942]            5.731        4.667  27   +1.064
(5.942, 6.504]            6.141        5.778  27   +0.364
(6.504, 8.103]            7.052        6.259  27   +0.793

Mean |bias|: 0.6272 corners per team per match
```

**Pre-committed decision rule applied:**
- CI excludes zero (full CI < 0): **NO** — CI = [−0.140, +0.097]
- Mean |bias| < 0.4 corners: **NO** — bias = 0.627

**DECISION: PARK.**

**Why the calibration fails:** the lowest-λ bucket shows systematic
under-prediction of ~1.35 corners. These are historical WC matches (1958-1990)
where `elo_final_fallback` assigns current Elo values to teams that no longer
exist as nations (Soviet Union, Czechoslovakia, West Germany) — those current
Elo values are very low (post-dissolution era), which drives λ down, but the
actual corner counts are normal. The 25 fallback matches are concentrated in the
low end of the λ distribution and poison that bucket. The fix would be to drop
those 25 entirely (they have no reliable Elo at match time); doing so would
leave 172 train matches — already a thin basis for a second model.

**Root constraint:** 197 men's training matches is the hard limit of what
StatsBomb makes freely available. At n=135 test matches, any real effect would
need to be large to clear a two-tailed 95% CI. A corners model that's merely
"not worse than the mean" is not useful to the API.

**To revisit if StatsBomb releases more event data** — the name-mapping
infrastructure (`MANUAL_MAP`, `try_map()`) and all evaluation machinery
are preserved in `scratchpad/corners_mens_only.py`. The 2 MANUAL_MAP entries
(`Congo DR → DR Congo`, `Côte d'Ivoire → Ivory Coast`) would need to be copied
into `predictor_core.py` if a future run integrates this.

## Phase 6.5 — Team form analysis (`get_team_form`)

**What it does:** `get_team_form(team_name, played_df, n=10, elo_final)` scans the
last `n` matches involving a team (home or away), computes per-match W/D/L and
goals, and returns:

```
result_string     "WWWWDLWWWD"  (oldest → newest, length = n_matches)
current_streak    int           consecutive non-losses counting backward from most recent
avg_goals_for     float
avg_goals_against float
points_per_game   float
current_elo       float
elo_n_ago         float         (Elo as of the oldest match in the window)
elo_trend         "rising"|"stable"|"falling"  (threshold ±20 pts)
last_5            list of {opponent, score, result, date}
n_matches         int
```

Also wired into `/api/form/<team>` endpoint, and into `/api/predictions` and
`/api/matchup` as `home_form` / `away_form` (last 5 of `result_string`, shown
as colour-coded W/D/L badges in the dashboard).

**Bug found and fixed — streak vs. rate:**

The first implementation computed `unbeaten_rate = wins + draws` (total
non-losses over the window) and generated "unbeaten in N of last M" phrasing
regardless of whether those non-losses were consecutive or scattered. This is
misleading: "unbeaten in 9 of last 10" strongly implies a current run, not just
a proportion.

England's last 10 at the time of validation:

```
result_string = "WWWWDLWWWD"   (oldest → newest)
wins + draws  = 9              (non-loss rate = 9/10)
current_streak = 4             (WWWD counting back from the right — blocked by the L)
```

Fix:
```python
current_streak = 0
for r in reversed(result_string):
    if r != "L":
        current_streak += 1
    else:
        break
```

Rule: use streak language ("on a N-game unbeaten run") **only** when
`current_streak == wins + draws` — i.e. all non-losses are genuinely
consecutive from the present. Otherwise use rate language ("XW-YD-ZL in their
last N"). This distinction prevents a team that lost three matches ago from
being described as "on an unbeaten run."

**Validation (literal output):**

```
Argentina  result_string=LWWWWWWWWW  streak=9  rate=9  → streak language ✓
England    result_string=WWWWDLWWWD  streak=4  rate=9  → rate language   ✓
```

## Phase 7 — Human-language match previews (`generate_match_preview`)

**What it does:** composes a 3–5 sentence natural-language preview from the
model's own outputs — no hardcoded facts, no external text.

Sentence structure:
1. Home team form (streak or rate language, Phase 6.5 rule)
2. Away team form (same)
3. H2H summary (last 5 meetings, or "No previous meetings recorded" if absent)
4. Elo gap narrative + model W/D/L% + most likely scoreline + its probability
5. (knockout matches only) Advancement % after ET and potential shootout

Added as `preview` field to both `/api/predictions` and `/api/matchup`. Also
wired into `index.html`: form badges (last-5 W/D/L letters, colour-coded
green/dim/red) shown above each prediction card; preview text block shown below
the scoreline heatmap in the matchup explorer.

**Validation — 3 cases (literal output as generated by the live model):**

*Lopsided — Jordan vs Argentina (neutral, not knockout):*
> Jordan arrive 3W-2D-5L in their last 10, averaging 1.4 scored and 1.8
> conceded per match. Argentina arrive on a 9-game unbeaten run, averaging 2.6
> scored and 0.2 conceded per match. No previous meetings between Jordan and
> Argentina are recorded in the dataset. The model gives Jordan a 1.9% win
> probability, draw 6.8%, Argentina 91.3%, with Argentina holding a 504-point
> Elo edge (2194 vs 1689); most likely scoreline is 0-3 (14.5%).

*Close — England vs France (neutral, not knockout):*
> England arrive 7W-2D-1L in their last 10, averaging 2.1 scored and 0.4
> conceded per match. France arrive 8W-1D-1L in their last 10, averaging 2.8
> scored and 1.0 conceded per match. In their last 5 meetings, England won 1,
> drawn 1, France won 3. The model gives England a 22.6% win probability, draw
> 25.9%, France 51.5%, with France holding a 98-point Elo edge (2159 vs 2062);
> most likely scoreline is 1-1 (12.3%).

*Knockout — Panama vs England (neutral, is_knockout=True):*
> Panama arrive 3W-3D-4L in their last 10, averaging 1.4 scored and 1.5
> conceded per match. England arrive 7W-2D-1L in their last 10, averaging 2.1
> scored and 0.4 conceded per match. In their last 1 meeting, England won 1.
> The model gives Panama a 8.5% win probability, draw 18.8%, England 72.7%,
> with England holding a 310-point Elo edge (2062 vs 1752); most likely
> scoreline is 0-2 (15.1%). Factoring in extra time and a potential penalty
> shootout, Panama have a 13.7% chance of advancing versus 86.3% for England.

Note: England's form sentence uses rate language ("7W-2D-1L in their last 10")
in all three cases — confirming the Phase 6.5 streak fix is active, since
England's current_streak (4) ≠ unbeaten_rate (9).

## Phase 8 — Extended form stats (StatsBomb coverage check → UI upgrade)

**Attempt:** per-team last-10-match averages for fouls, cards, passes, throw-ins, and corners, extending `get_team_form()` using the StatsBomb open-data fetch and team-name-mapping infrastructure from Phase 6.

**Pre-committed gate:** average StatsBomb coverage across the 12 current fixture teams must be ≥ 4/10 matches before building anything user-facing.

**Result: 0/10 for every team. Gate not met — stopped as committed.**

The root cause is structural: StatsBomb's open-data repository ends at the 2024 Copa América (through 15 July 2024). Every one of the 12 current teams' last-10 matches falls between October 2025 and June 2026 — entirely outside the available window. The wrong fuzzy matches surfaced during the name-mapping pass illustrate concretely why the gate exists: Austria matched to "Australia," Armenia matched to "Argentina," Faroe Islands matched to "Cape Verde Islands," and Jordan/Uzbekistan matched nothing at all. Even if a handful of 2025–26 matches had appeared in the index, these mis-mappings would have silently poisoned the coverage count.

**What was built instead:** `get_team_form()` already returns `avg_goals_for`, `avg_goals_against`, `points_per_game`, and `elo_trend` with full coverage (results.csv, all teams, all matches). These were surfaced in the UI: both `/api/predictions` and `/api/matchup` now include `home_form_stats` / `away_form_stats` dicts, and the dashboard replaces the previous single-row "Form: WWLDL" badge with a two-column block showing the W/D/L string plus trend arrow (↑/→/↓), GF/GA averages, and points-per-game — all drawn from full-coverage data, no caveats needed.

## Files in this delivery
- `predictor_core.py` — full pipeline (data, Elo, features, model, Dixon-Coles,
  knockout/shootout layer, SHAP explanations)
- `server.py` — Flask backend, auto-refresh, API
- `static/index.html` — dashboard UI
- `worldcup_predictor.py` — CLI report version
- `requirements.txt`, `UI_README.md`