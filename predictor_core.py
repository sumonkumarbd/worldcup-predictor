"""
predictor_core.py — shared model logic for the World Cup predictor.
Used by both worldcup_predictor.py (CLI/report) and server.py (web UI backend).
"""
import os
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import shap

# ============================================================
# CONFIG
# ============================================================
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
LOCAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

PATCH_RESULTS = [
    ("Egypt", "Iran", 1, 1),
    ("New Zealand", "Belgium", 1, 5),
    ("Cape Verde", "Saudi Arabia", 0, 0),
    ("Uruguay", "Spain", 0, 1),
    ("Norway", "France", 1, 4),
    ("Senegal", "Iraq", 5, 0),
]

MARKET_ODDS = {
    ("Panama", "England"):       dict(H=5.4,  D=10.7, A=83.9),
    ("Croatia", "Ghana"):        dict(H=50.5, D=29.9, A=19.6),
    ("Colombia", "Portugal"):    dict(H=26.5, D=25.3, A=48.2),
    ("DR Congo", "Uzbekistan"):  dict(H=60.4, D=23.0, A=16.6),
    ("Jordan", "Argentina"):     dict(H=4.8,  D=11.2, A=84.0),
    ("Algeria", "Austria"):      dict(H=22.1, D=46.0, A=31.9),
}

# 2026 World Cup: last group-stage matches in the source CSV are dated 2026-06-26;
# every FIFA World Cup match from 2026-06-27 onward is a knockout fixture.
# (The martj42 repo dates the Round of 32 as Jun 27, one day earlier than the
# official Jun 28 start — no group-stage match in the dataset falls on Jun 27.)
KNOCKOUT_START_DATE = pd.Timestamp('2026-06-27')

# Shootout model: fitted on 677 historical shootouts via logistic regression
# shootout_home_win ~ intercept + coef * elo_diff_pre
# 95% CI for coef: [0.000520, 0.002775], p=0.0042 — Elo is significant.
SHOOTOUT_INTERCEPT = 0.102443
SHOOTOUT_COEF = 0.001647

TRAIN_START, TRAIN_END = "2000-01-01", "2023-12-31"
TEST_START = "2024-01-01"
GMEAN = 1.4
ELO_INIT = 1500.0
HOME_ADV_ELO = 100
MAJOR_CONTINENTAL = ["Euro", "Copa América", "Copa America", "African Cup of Nations",
                     "AFCON", "AFC Asian Cup", "Gold Cup", "CONCACAF Championship",
                     "Confederations Cup"]
SCORE_CAP = 35
FEATS = ["elo_diff", "team_gf", "team_ga", "opp_gf", "opp_ga", "h2h_win_rate", "h2h_draw_rate", "h2h_avg_goal_diff", "team_days_since", "opp_days_since", "is_home_adv", "k"]
PASS_CRITERIA = dict(max_calibration_err=0.08, beat_floor_pct=0.03)


# ============================================================
# DATA + FEATURES
# ============================================================
def load_data(force_refresh=False):
    if force_refresh or not os.path.exists(LOCAL_CSV):
        df = pd.read_csv(DATA_URL)
        df.to_csv(LOCAL_CSV, index=False)
    df = pd.read_csv(LOCAL_CSV)
    df["date"] = pd.to_datetime(df["date"])
    for home, away, hs, asc in PATCH_RESULTS:
        mask = (df["home_team"] == home) & (df["away_team"] == away) & (df["home_score"].isna())
        df.loc[mask, "home_score"] = hs
        df.loc[mask, "away_score"] = asc
    return df.sort_values("date").reset_index(drop=True)


def k_factor(tournament):
    if tournament == "FIFA World Cup":
        return 60
    if "World Cup qualification" in tournament:
        return 35
    if any(m in tournament for m in MAJOR_CONTINENTAL) and "qualification" not in tournament:
        return 40
    if "qualification" in tournament:
        return 30
    if tournament == "Friendly":
        return 20
    return 20


def build_features(df):
    """One chronological pass -> (per-match feature rows, final elo dict, final form dict)."""
    df = df.copy()
    df["k"] = df["tournament"].apply(k_factor)

    elo = defaultdict(lambda: ELO_INIT)
    recent_gf, recent_ga = defaultdict(lambda: deque(maxlen=10)), defaultdict(lambda: deque(maxlen=10))
    h2h = defaultdict(lambda: deque(maxlen=5))
    last_played = defaultdict(lambda: None)
    rows = []

    for _, r in df.iterrows():
        h, a = r["home_team"], r["away_team"]
        home_last = last_played[h]
        away_last = last_played[a]
        home_days_since = (r["date"] - home_last).days if home_last is not None else np.nan
        away_days_since = (r["date"] - away_last).days if away_last is not None else np.nan
        key = frozenset({h, a})
        pair_history = h2h[key]
        if pair_history:
            total = len(pair_history)
            home_wins = sum(1 for hs, ascore in pair_history if hs > ascore)
            draws = sum(1 for hs, ascore in pair_history if hs == ascore)
            away_wins = total - home_wins - draws
            home_h2h_win_rate = home_wins / total
            home_h2h_draw_rate = draws / total
            home_h2h_avg_goal_diff = sum(hs - ascore for hs, ascore in pair_history) / total
            away_h2h_win_rate = away_wins / total
            away_h2h_draw_rate = draws / total
            away_h2h_avg_goal_diff = sum(ascore - hs for hs, ascore in pair_history) / total
        else:
            home_h2h_win_rate = home_h2h_draw_rate = away_h2h_win_rate = away_h2h_draw_rate = np.nan
            home_h2h_avg_goal_diff = away_h2h_avg_goal_diff = np.nan
        eh, ea = elo[h], elo[a]
        rows.append({
            "date": r["date"], "home_team": h, "away_team": a,
            "home_elo_pre": eh, "away_elo_pre": ea,
            "home_gf_pre": np.mean(recent_gf[h]) if recent_gf[h] else np.nan,
            "home_ga_pre": np.mean(recent_ga[h]) if recent_ga[h] else np.nan,
            "away_gf_pre": np.mean(recent_gf[a]) if recent_gf[a] else np.nan,
            "away_ga_pre": np.mean(recent_ga[a]) if recent_ga[a] else np.nan,
            "home_days_since": home_days_since,
            "away_days_since": away_days_since,
            "home_h2h_win_rate": home_h2h_win_rate,
            "home_h2h_draw_rate": home_h2h_draw_rate,
            "home_h2h_avg_goal_diff": home_h2h_avg_goal_diff,
            "away_h2h_win_rate": away_h2h_win_rate,
            "away_h2h_draw_rate": away_h2h_draw_rate,
            "away_h2h_avg_goal_diff": away_h2h_avg_goal_diff,
            "neutral": r["neutral"], "tournament": r["tournament"], "k": r["k"],
            "home_score": r["home_score"], "away_score": r["away_score"],
        })
        last_played[h] = r["date"]
        last_played[a] = r["date"]
        if pd.isna(r["home_score"]):
            continue
        hs, asc = r["home_score"], r["away_score"]
        diff = abs(hs - asc)
        g = 1.0 if diff <= 1 else (1.5 if diff == 2 else (11 + diff) / 8)
        adv = 0 if r["neutral"] else HOME_ADV_ELO
        we_h = 1 / (10 ** (-((eh + adv) - ea) / 400) + 1)
        w_h = 1.0 if hs > asc else (0.5 if hs == asc else 0.0)
        elo[h] = eh + r["k"] * g * (w_h - we_h)
        elo[a] = ea + r["k"] * g * ((1 - w_h) - (1 - we_h))
        recent_gf[h].append(hs); recent_ga[h].append(asc)
        recent_gf[a].append(asc); recent_ga[a].append(hs)
        h2h[key].append((hs, asc))

    form_final = {t: (np.mean(recent_gf[t]) if recent_gf[t] else GMEAN,
                       np.mean(recent_ga[t]) if recent_ga[t] else GMEAN)
                  for t in set(list(elo.keys()))}
    return pd.DataFrame(rows), dict(elo), form_final


def to_long(d):
    home = pd.DataFrame({
        "team_elo": d["home_elo_pre"], "opp_elo": d["away_elo_pre"],
        "team_gf": d["home_gf_pre"].fillna(GMEAN), "team_ga": d["home_ga_pre"].fillna(GMEAN),
        "opp_gf": d["away_gf_pre"].fillna(GMEAN), "opp_ga": d["away_ga_pre"].fillna(GMEAN),
        "h2h_win_rate": d["home_h2h_win_rate"].fillna(0.33),
        "h2h_draw_rate": d["home_h2h_draw_rate"].fillna(0.33),
        "h2h_avg_goal_diff": d["home_h2h_avg_goal_diff"].fillna(0.0),
        "team_days_since": d["home_days_since"].fillna(7),
        "opp_days_since": d["away_days_since"].fillna(7),
        "is_home_adv": np.where(d["neutral"], 0, 1), "k": d["k"], "goals": d["home_score"],
    })
    away = pd.DataFrame({
        "team_elo": d["away_elo_pre"], "opp_elo": d["home_elo_pre"],
        "team_gf": d["away_gf_pre"].fillna(GMEAN), "team_ga": d["away_ga_pre"].fillna(GMEAN),
        "opp_gf": d["home_gf_pre"].fillna(GMEAN), "opp_ga": d["home_ga_pre"].fillna(GMEAN),
        "h2h_win_rate": d["away_h2h_win_rate"].fillna(0.33),
        "h2h_draw_rate": d["away_h2h_draw_rate"].fillna(0.33),
        "h2h_avg_goal_diff": d["away_h2h_avg_goal_diff"].fillna(0.0),
        "team_days_since": d["away_days_since"].fillna(7),
        "opp_days_since": d["home_days_since"].fillna(7),
        "is_home_adv": 0, "k": d["k"], "goals": d["away_score"],
    })
    out = pd.concat([home, away], ignore_index=True)
    out["elo_diff"] = out["team_elo"] - out["opp_elo"]
    return out


# ============================================================
# MODEL
# ============================================================
def train_model(played):
    train_m = played[(played.date >= TRAIN_START) & (played.date <= TRAIN_END)]
    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                         max_depth=4, min_samples_leaf=40, random_state=42)
    gbm.fit(to_long(train_m)[FEATS], to_long(train_m)["goals"])
    return gbm


def predict_lambdas(gbm, d):
    long_d = to_long(d)
    lam = gbm.predict(long_d[FEATS])
    n = len(d)
    return lam[:n], lam[n:]


def fit_rho(played, gbm):
    train_m = played[(played.date >= TRAIN_START) & (played.date <= TRAIN_END)]
    lam_h, lam_a = predict_lambdas(gbm, train_m)
    y_h = train_m.home_score.values.astype(int)
    y_a = train_m.away_score.values.astype(int)

    def neg_ll(rho):
        rho = float(rho)
        total = 0.0
        for lh, la, hs, ascore in zip(lam_h, lam_a, y_h, y_a):
            M = scoreline_matrix(lh, la, rho=rho)
            total -= np.log(np.clip(M[hs, ascore], 1e-15, None))
        return total

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.0), method='bounded', options={'xatol': 1e-4})
    rho = float(res.x) if res.success else 0.0
    return rho


def scoreline_matrix(lh, la, rho=0.0, cap=SCORE_CAP):
    i = np.arange(cap + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    if rho != 0.0:
        M[0, 0] *= 1 - lh * la * rho
        M[1, 0] *= 1 + la * rho
        M[0, 1] *= 1 + lh * rho
        M[1, 1] *= 1 - rho
    return M / M.sum()


def outcome_probs(M):
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def safe_log_loss(y_true_arr, prob_arr, col_order):
    idx = {c: i for i, c in enumerate(col_order)}
    p = np.array([prob_arr[n, idx[y]] for n, y in enumerate(y_true_arr)])
    return -np.mean(np.log(np.clip(p, 1e-15, 1)))


def shootout_home_prob(elo_diff_pre):
    """P(home team wins penalty shootout) from Elo-based logistic model."""
    logit = SHOOTOUT_INTERCEPT + SHOOTOUT_COEF * elo_diff_pre
    return 1.0 / (1.0 + np.exp(-logit))


def advancement_probability(lam_home, lam_away, elo_diff_pre, rho=0.0):
    """
    P(home/away advances) in a knockout match.
    90 min → ET (lambdas scaled by 30/90) → penalty shootout.
    Returns (p_home_advances, p_away_advances); asserts they sum to 1.
    """
    M90 = scoreline_matrix(lam_home, lam_away, rho=rho)
    p_home_90, p_draw_90, p_away_90 = outcome_probs(M90)

    M_et = scoreline_matrix(lam_home / 3.0, lam_away / 3.0, rho=rho)
    p_home_et, p_draw_et, p_away_et = outcome_probs(M_et)

    p_sh = shootout_home_prob(elo_diff_pre)
    p_home_adv = p_home_90 + p_draw_90 * (p_home_et + p_draw_et * p_sh)
    p_away_adv = p_away_90 + p_draw_90 * (p_away_et + p_draw_et * (1.0 - p_sh))

    assert abs(p_home_adv + p_away_adv - 1.0) < 1e-9, \
        f"advancement_probability: probs sum to {p_home_adv + p_away_adv}"
    return p_home_adv, p_away_adv


def predict_matchup(gbm, elo_final, form_final, home, away, neutral=True, k=60, rho=0.0, is_knockout=False):
    """Predict any arbitrary matchup using each team's CURRENT rating/form."""
    eh, ea = elo_final.get(home, ELO_INIT), elo_final.get(away, ELO_INIT)
    gf_h, ga_h = form_final.get(home, (GMEAN, GMEAN))
    gf_a, ga_a = form_final.get(away, (GMEAN, GMEAN))
    is_home_adv = 0 if neutral else 1
    Xh = pd.DataFrame([{"elo_diff": eh - ea, "team_gf": gf_h, "team_ga": ga_h,
                         "opp_gf": gf_a, "opp_ga": ga_a, "h2h_win_rate": 0.33,
                         "h2h_draw_rate": 0.33, "h2h_avg_goal_diff": 0.0,
                         "team_days_since": 7, "opp_days_since": 7,
                         "is_home_adv": is_home_adv, "k": k}])
    Xa = pd.DataFrame([{"elo_diff": ea - eh, "team_gf": gf_a, "team_ga": ga_a,
                         "opp_gf": gf_h, "opp_ga": ga_h, "h2h_win_rate": 0.33,
                         "h2h_draw_rate": 0.33, "h2h_avg_goal_diff": 0.0,
                         "team_days_since": 7, "opp_days_since": 7,
                         "is_home_adv": 0, "k": k}])
    lam_h = float(gbm.predict(Xh[FEATS])[0])
    lam_a = float(gbm.predict(Xa[FEATS])[0])
    M = scoreline_matrix(lam_h, lam_a, rho=rho)
    ph, pdraw, pa = outcome_probs(M)
    i, j = np.unravel_index(np.argmax(M), M.shape)
    result = dict(home=home, away=away, neutral=neutral, lam_home=round(lam_h, 2), lam_away=round(lam_a, 2),
                  home_elo=round(eh, 1), away_elo=round(ea, 1),
                  p_home=round(ph*100, 1), p_draw=round(pdraw*100, 1), p_away=round(pa*100, 1),
                  most_likely_score=f"{i}-{j}", p_most_likely=round(float(M[i, j])*100, 1),
                  matrix=M[:6, :6].round(4).tolist())
    if is_knockout:
        elo_diff_pre = (eh - ea) if neutral else (eh + HOME_ADV_ELO - ea)
        p_home_adv, p_away_adv = advancement_probability(lam_h, lam_a, elo_diff_pre, rho=rho)
        result["is_knockout"] = True
        result["p_home_advances"] = round(p_home_adv * 100, 1)
        result["p_away_advances"] = round(p_away_adv * 100, 1)
    return result


FEATURE_EXPLANATIONS = {
    'elo_diff': ('Elo differential', 'higher Elo than opponent', 'lower Elo than opponent'),
    'team_gf': ('recent scoring form', 'strong scoring form', 'weak scoring form'),
    'team_ga': ('recent conceding form', 'low goals conceded', 'high goals conceded'),
    'opp_gf': ('opponent attacking strength', 'opponent weak attack', 'opponent strong attack'),
    'opp_ga': ('opponent defensive weakness', 'opponent weak defense', 'opponent strong defense'),
    'h2h_win_rate': ('head-to-head win rate', 'better head-to-head record', 'worse head-to-head record'),
    'h2h_draw_rate': ('head-to-head draw rate', 'higher draw tendency in prior meetings', 'lower draw tendency in prior meetings'),
    'h2h_avg_goal_diff': ('head-to-head goal difference', 'strong goal difference vs this opponent', 'weak goal difference vs this opponent'),
    'team_days_since': ('rest days', 'more rest than usual', 'less rest than usual'),
    'opp_days_since': ('opponent rest days', 'opponent more rested', 'opponent less rested'),
    'is_home_adv': ('home advantage', 'home advantage present', 'home advantage absent'),
    'k': ('tournament importance', 'higher tournament importance', 'lower tournament importance'),
}

RAW_DIRECTIONAL_FEATURES = {
    'elo_diff', 'is_home_adv', 'team_gf', 'team_ga', 'opp_gf', 'opp_ga',
    'team_days_since', 'opp_days_since'
}

RAW_DIRECTIONAL_BASELINES = {
    'elo_diff': lambda x: x > 0,
    'is_home_adv': lambda x: x > 0,
    'team_gf': lambda x: x >= GMEAN,
    'team_ga': lambda x: x <= GMEAN,
    'opp_gf': lambda x: x <= GMEAN,
    'opp_ga': lambda x: x >= GMEAN,
    'team_days_since': lambda x: x >= 7,
    'opp_days_since': lambda x: x >= 7,
}


def _shap_pct_effect(impact):
    return float(np.expm1(impact) * 100.0)


def _raw_direction_positive(feature, raw_value):
    if feature not in RAW_DIRECTIONAL_BASELINES or raw_value is None:
        return None
    return bool(RAW_DIRECTIONAL_BASELINES[feature](raw_value))


def _feature_contradicts_raw_direction(feature, impact, raw_value):
    raw_positive = _raw_direction_positive(feature, raw_value)
    if raw_positive is None:
        return False
    if impact == 0:
        return False
    return (impact > 0) != raw_positive


def _feature_reason(feature, impact, raw_value=None):
    label, positive_phrase, negative_phrase = FEATURE_EXPLANATIONS.get(feature, (feature, feature, feature))
    pct = _shap_pct_effect(impact)
    fmt = f'{pct:+.0f}%'
    raw_positive = _raw_direction_positive(feature, raw_value)
    if raw_positive is not None:
        return f'{positive_phrase} ({fmt})' if raw_positive else f'{negative_phrase} ({fmt})'
    if feature == 'is_home_adv':
        return f'{positive_phrase} ({fmt})' if impact > 0 else f'{negative_phrase} ({fmt})'
    if impact >= 0:
        return f'{positive_phrase} ({fmt})'
    return f'{negative_phrase} ({fmt})'


def _build_matchup_frames(home, away, neutral, k, elo_final, form_final, played=None):
    eh, ea = elo_final.get(home, ELO_INIT), elo_final.get(away, ELO_INIT)
    fh = form_final.get(home, (GMEAN, GMEAN))
    fa = form_final.get(away, (GMEAN, GMEAN))
    is_home_adv = 0 if neutral else 1

    # Use real H2H when played DataFrame is available; fall back to neutral
    # defaults only for pairs with zero recorded history.
    # team_days_since is left at 7 in both cases — genuinely unknowable for
    # a hypothetical or future fixture without a confirmed kickoff date.
    if played is not None:
        h2h_data = _get_h2h(home, away, played, n=5)
        n_h2h = h2h_data["n_meetings"]
        if n_h2h > 0:
            hw, dr = h2h_data["home_wins"], h2h_data["draws"]
            h2h_wr_h = hw / n_h2h
            h2h_dr   = dr / n_h2h
            h2h_gd_h = sum(m["home_score"] - m["away_score"]
                           for m in h2h_data["meetings"]) / n_h2h
        else:
            h2h_wr_h = h2h_dr = 0.33
            h2h_gd_h = 0.0
    else:
        h2h_wr_h = h2h_dr = 0.33
        h2h_gd_h = 0.0

    Xh = pd.DataFrame([{
        'elo_diff': eh - ea,
        'team_gf': fh[0], 'team_ga': fh[1],
        'opp_gf': fa[0], 'opp_ga': fa[1],
        'h2h_win_rate': h2h_wr_h, 'h2h_draw_rate': h2h_dr,
        'h2h_avg_goal_diff': h2h_gd_h,
        'team_days_since': 7, 'opp_days_since': 7,
        'is_home_adv': is_home_adv, 'k': k
    }])
    Xa = pd.DataFrame([{
        'elo_diff': ea - eh,
        'team_gf': fa[0], 'team_ga': fa[1],
        'opp_gf': fh[0], 'opp_ga': fh[1],
        'h2h_win_rate': 1 - h2h_wr_h - h2h_dr,   # away's win rate
        'h2h_draw_rate': h2h_dr,
        'h2h_avg_goal_diff': -h2h_gd_h,            # flip perspective
        'team_days_since': 7, 'opp_days_since': 7,
        'is_home_adv': 0, 'k': k
    }])
    return Xh, Xa


def explain_matchup(gbm, elo_final, form_final, home, away, neutral=True, k=60, rho=0.0, n_top=3):
    Xh, Xa = _build_matchup_frames(home, away, neutral, k, elo_final, form_final)
    explainer = shap.TreeExplainer(gbm)
    shap_h_vals = explainer.shap_values(Xh[FEATS])
    shap_a_vals = explainer.shap_values(Xa[FEATS])
    shap_h = np.array(shap_h_vals).reshape(-1)
    shap_a = np.array(shap_a_vals).reshape(-1)
    base_value = float(explainer.expected_value if np.ndim(explainer.expected_value) == 0 else np.asarray(explainer.expected_value).reshape(-1)[0])
    lam_h = float(gbm.predict(Xh[FEATS])[0])
    lam_a = float(gbm.predict(Xa[FEATS])[0])

    def format_explanation(shap_vals, X, side):
        shap_arr = np.array(shap_vals).reshape(-1)
        raw_row = X.iloc[0].to_dict()
        impacts = [(feature, shap_arr[idx], raw_row) for idx, feature in enumerate(FEATS)]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        filtered = []
        for feature, impact, raw_row in impacts:
            raw_value = raw_row.get(feature)
            if _feature_contradicts_raw_direction(feature, impact, raw_value):
                continue
            filtered.append((feature, impact, raw_row))
            if len(filtered) >= n_top:
                break
        top_impacts = filtered[:n_top]
        parts = [_feature_reason(f, v, raw_row.get(f)) for f, v, raw_row in top_impacts]
        score = lam_h if side == 'Home' else lam_a
        return f"{side} expected goals {score:.2f}: " + ", ".join(parts)

    home_explanation = format_explanation(shap_h, Xh, 'Home')
    away_explanation = format_explanation(shap_a, Xa, 'Away')

    home_shap_pct = {f: float(np.expm1(v) * 100.0) for f, v in zip(FEATS, shap_h)}
    away_shap_pct = {f: float(np.expm1(v) * 100.0) for f, v in zip(FEATS, shap_a)}

    return {
        'home_expected_goals': round(lam_h, 2),
        'away_expected_goals': round(lam_a, 2),
        'rho': rho,
        'home_shap': home_shap_pct,
        'away_shap': away_shap_pct,
        'home_explanation': home_explanation,
        'away_explanation': away_explanation,
        'base_value': base_value,
    }


def validate_shap_linkage(gbm, elo_final, form_final, matchups=None, tol=1e-3):
    if matchups is None:
        matchups = [
            ('England', 'Germany', False),
            ('Brazil', 'Argentina', False),
            ('Japan', 'Spain', True),
            ('France', 'Spain', False),
            ('USA', 'Mexico', False),
        ]
    explainer = shap.TreeExplainer(gbm)
    base_value = explainer.expected_value if np.ndim(explainer.expected_value) == 0 else np.asarray(explainer.expected_value).reshape(-1)[0]
    for home, away, neutral in matchups:
        Xh, Xa = _build_matchup_frames(home, away, neutral, 60, elo_final, form_final)
        for X in (Xh, Xa):
            shap_vals = np.array(explainer.shap_values(X[FEATS])).reshape(-1)
            predicted = float(gbm.predict(X[FEATS])[0])
            reconstructed = float(np.exp(base_value + shap_vals.sum()))
            if abs(reconstructed - predicted) >= tol:
                raise AssertionError(
                    f'SHAP linkage failed for {home}/{away} neutral={neutral}: '
                    f'pred={predicted:.6f} reconstructed={reconstructed:.6f} diff={abs(reconstructed-predicted):.6f}'
                )
    return True


def _predicate_phrase_positive(feature, phrase):
    _, positive_phrase, negative_phrase = FEATURE_EXPLANATIONS.get(feature, (feature, feature, feature))
    return positive_phrase in phrase


def _select_top_explanation_features(gbm, X, n_top=3):
    explainer = shap.TreeExplainer(gbm)
    shap_vals = np.array(explainer.shap_values(X[FEATS])).reshape(-1)
    raw_row = X.iloc[0].to_dict()
    impacts = [(feature, shap_vals[idx], raw_row) for idx, feature in enumerate(FEATS)]
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    selected = []
    for feature, impact, raw_row in impacts:
        raw_value = raw_row.get(feature)
        if _feature_contradicts_raw_direction(feature, impact, raw_value):
            continue
        selected.append((feature, impact, raw_row))
        if len(selected) >= n_top:
            break
    return selected


def validate_explanation_consistency(gbm, elo_final, form_final, teams, upcoming=None, n_random=30, seed=42):
    if upcoming is None:
        upcoming = []
    samples = []
    samples.extend(upcoming)
    random.seed(seed)
    seen = set((h, a) for h, a, _, _ in upcoming)
    team_list = list(teams)
    while len(samples) < len(upcoming) + n_random:
        h, a = random.sample(team_list, 2)
        key = (h, a)
        if key in seen:
            continue
        seen.add(key)
        samples.append((h, a, True, 60))
    violations = []
    for home, away, neutral, k in samples:
        Xh, Xa = _build_matchup_frames(home, away, neutral, k, elo_final, form_final)
        for side, X in [('Home', Xh), ('Away', Xa)]:
            top = _select_top_explanation_features(gbm, X, n_top=3)
            for feature, impact, raw_row in top:
                portions = _feature_reason(feature, impact, raw_row.get(feature))
                phrase_positive = _predicate_phrase_positive(feature, portions)
                impact_positive = impact > 0
                if impact != 0 and phrase_positive != impact_positive:
                    violations.append({
                        'home': home,
                        'away': away,
                        'neutral': neutral,
                        'k': k,
                        'side': side,
                        'feature': feature,
                        'raw_value': raw_row.get(feature),
                        'impact': impact,
                        'shap_pct': _shap_pct_effect(impact),
                        'phrase': portions,
                        'phrase_positive': phrase_positive,
                        'impact_positive': impact_positive,
                    })
    return violations


# ============================================================
# BACKTEST
# ============================================================
def run_backtest(gbm, played, rho=0.0):
    test_m = played[played.date >= TEST_START]
    classes = ["H", "D", "A"]
    y_true = np.select([test_m.home_score > test_m.away_score, test_m.home_score == test_m.away_score],
                        ["H", "D"], default="A")

    lam_h, lam_a = predict_lambdas(gbm, test_m)
    probs = np.array([outcome_probs(scoreline_matrix(lh, la, rho=rho)) for lh, la in zip(lam_h, lam_a)])
    ll_model = safe_log_loss(y_true, probs, classes)

    train_m = played[(played.date >= TRAIN_START) & (played.date <= TRAIN_END)]
    elo_diff_tr = np.where(train_m.neutral, train_m.home_elo_pre - train_m.away_elo_pre,
                            (train_m.home_elo_pre + HOME_ADV_ELO) - train_m.away_elo_pre)
    elo_diff_te = np.where(test_m.neutral, test_m.home_elo_pre - test_m.away_elo_pre,
                            (test_m.home_elo_pre + HOME_ADV_ELO) - test_m.away_elo_pre)
    y_train = np.select([train_m.home_score > train_m.away_score, train_m.home_score == train_m.away_score],
                         ["H", "D"], default="A")
    base_clf = LogisticRegression(max_iter=1000).fit(elo_diff_tr.reshape(-1, 1), y_train)
    base_probs = base_clf.predict_proba(elo_diff_te.reshape(-1, 1))
    ll_base = safe_log_loss(y_true, base_probs, list(base_clf.classes_))

    rate = pd.Series(y_train).value_counts(normalize=True).reindex(classes).values
    ll_floor = safe_log_loss(y_true, np.tile(rate, (len(y_true), 1)), classes)

    p_home = probs[:, 0]
    actual_home_win = (test_m.home_score > test_m.away_score).astype(int).values
    bins = pd.qcut(p_home, 10, duplicates="drop")
    cal = pd.DataFrame({"bin": bins, "pred": p_home, "actual": actual_home_win})
    cal_table = cal.groupby("bin").agg(pred_mean=("pred", "mean"), actual_rate=("actual", "mean"), n=("actual", "size"))
    cal_err = float((cal_table["pred_mean"] - cal_table["actual_rate"]).abs().mean())

    idx_h = {c: i for i, c in enumerate(classes)}
    per_ll_model = -np.log(np.clip([probs[n, idx_h[y]] for n, y in enumerate(y_true)], 1e-15, 1))
    idx_b = {c: i for i, c in enumerate(base_clf.classes_)}
    per_ll_base = -np.log(np.clip([base_probs[n, idx_b[y]] for n, y in enumerate(y_true)], 1e-15, 1))
    rng = np.random.default_rng(42)
    diff = per_ll_model - per_ll_base
    boot = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(4000)]
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    crit_a = bool(ci_hi < 0)  # requires the FULL bootstrap CI to be below zero, not just a point-estimate win
    crit_b = bool(cal_err <= PASS_CRITERIA["max_calibration_err"])
    crit_c = bool(ll_model <= ll_floor * (1 - PASS_CRITERIA["beat_floor_pct"]))

    return dict(n_test=len(test_m), ll_model=round(ll_model, 4), ll_base=round(ll_base, 4),
                ll_floor=round(ll_floor, 4), cal_err=round(cal_err, 4),
                ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
                gap_significant=not (ci_lo <= 0 <= ci_hi),
                crit_a=crit_a, crit_b=crit_b, crit_c=crit_c, overall=crit_a and crit_b and crit_c,
                cal_table=[{"pred_mean": round(r.pred_mean, 3), "actual_rate": round(r.actual_rate, 3), "n": int(r.n)}
                           for r in cal_table.itertuples()])


def predict_future(gbm, feat, rho=0.0):
    future = feat[feat.home_score.isna()].copy()
    if future.empty:
        return []
    lam_h, lam_a = predict_lambdas(gbm, future)
    out = []
    for (_, row), lh, la in zip(future.iterrows(), lam_h, lam_a):
        M = scoreline_matrix(lh, la, rho=rho)
        ph, pdraw, pa = outcome_probs(M)
        i, j = np.unravel_index(np.argmax(M), M.shape)
        key = (row.home_team, row.away_team)
        mkt = MARKET_ODDS.get(key, {})
        entry = dict(
            home=row.home_team, away=row.away_team,
            lam_home=round(float(lh), 2), lam_away=round(float(la), 2),
            home_elo=round(float(row.home_elo_pre), 1),
            away_elo=round(float(row.away_elo_pre), 1),
            model_H=round(ph*100, 1), model_D=round(pdraw*100, 1), model_A=round(pa*100, 1),
            market_H=mkt.get("H"), market_D=mkt.get("D"), market_A=mkt.get("A"),
            most_likely_score=f"{i}-{j}", p_most_likely=round(float(M[i, j])*100, 1),
            neutral=bool(row.neutral), k=int(row.k),
            matrix=M[:6, :6].round(4).tolist(),
        )
        is_knockout = (row.tournament == 'FIFA World Cup') and (row.date >= KNOCKOUT_START_DATE)
        if is_knockout:
            neutral = bool(row.neutral)
            elo_diff_pre = (row.home_elo_pre - row.away_elo_pre) if neutral \
                else (row.home_elo_pre + HOME_ADV_ELO - row.away_elo_pre)
            p_home_adv, p_away_adv = advancement_probability(lh, la, elo_diff_pre, rho=rho)
            entry["is_knockout"] = True
            entry["p_home_advances"] = round(p_home_adv * 100, 1)
            entry["p_away_advances"] = round(p_away_adv * 100, 1)
        out.append(entry)
    return out


# ============================================================
# FORM ANALYSIS + H2H + MATCH PREVIEW
# ============================================================
def _get_h2h(home, away, played, n=5):
    mask = (
        ((played["home_team"] == home) & (played["away_team"] == away)) |
        ((played["home_team"] == away) & (played["away_team"] == home))
    )
    h2h = played[mask].sort_values("date").tail(n)
    if h2h.empty:
        return {"n_meetings": 0, "home_wins": 0, "draws": 0, "away_wins": 0, "meetings": []}
    home_wins = draws = away_wins = 0
    meetings = []
    for _, row in h2h.iterrows():
        if row["home_team"] == home:
            hs, as_ = int(row["home_score"]), int(row["away_score"])
        else:
            hs, as_ = int(row["away_score"]), int(row["home_score"])
        if hs > as_:
            home_wins += 1
        elif hs == as_:
            draws += 1
        else:
            away_wins += 1
        meetings.append({"date": row["date"].strftime("%Y-%m-%d"), "home_score": hs, "away_score": as_})
    return {"n_meetings": len(h2h), "home_wins": home_wins, "draws": draws, "away_wins": away_wins, "meetings": meetings}


def get_team_form(team_name, played_df, n=10, elo_final=None):
    """Return form stats for the n most recent matches involving team_name."""
    mask = (played_df["home_team"] == team_name) | (played_df["away_team"] == team_name)
    team_matches = played_df[mask].sort_values("date").tail(n)

    current_elo = round(float(elo_final.get(team_name, ELO_INIT)), 1) if elo_final else None

    if team_matches.empty:
        return {
            "result_string": "", "avg_goals_for": None, "avg_goals_against": None,
            "points_per_game": None, "current_elo": current_elo, "elo_n_ago": None,
            "elo_trend": "stable", "n_matches": 0, "last_5": [],
        }

    records = []
    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team_name
        if is_home:
            gf, ga = int(row["home_score"]), int(row["away_score"])
            opponent, venue = row["away_team"], ("N" if row["neutral"] else "H")
            team_elo_pre = float(row["home_elo_pre"])
        else:
            gf, ga = int(row["away_score"]), int(row["home_score"])
            opponent, venue = row["home_team"], ("N" if row["neutral"] else "A")
            team_elo_pre = float(row["away_elo_pre"])
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        records.append({"opponent": opponent, "venue": venue, "goals_for": gf, "goals_against": ga,
                        "result": result, "date": row["date"].strftime("%Y-%m-%d"),
                        "score": f"{gf}-{ga}", "team_elo_pre": team_elo_pre})

    result_string = "".join(r["result"] for r in records)
    avg_gf = round(float(np.mean([r["goals_for"] for r in records])), 2)
    avg_ga = round(float(np.mean([r["goals_against"] for r in records])), 2)
    pts = {"W": 3, "D": 1, "L": 0}
    ppg = round(float(np.mean([pts[r["result"]] for r in records])), 2)

    elo_n_ago = round(records[0]["team_elo_pre"], 1)
    elo_now = current_elo if current_elo is not None else round(records[-1]["team_elo_pre"], 1)
    elo_diff = elo_now - elo_n_ago
    trend = "rising" if elo_diff > 20 else ("falling" if elo_diff < -20 else "stable")

    # Consecutive non-losses counting backward from most recent match
    current_streak = 0
    for r in reversed(result_string):
        if r != "L":
            current_streak += 1
        else:
            break

    last_5 = [{"opponent": r["opponent"], "score": r["score"], "result": r["result"], "date": r["date"]}
               for r in records[-5:]]

    return {
        "result_string": result_string, "avg_goals_for": avg_gf, "avg_goals_against": avg_ga,
        "points_per_game": ppg, "current_elo": elo_now, "elo_n_ago": elo_n_ago,
        "elo_trend": trend, "n_matches": len(records), "current_streak": current_streak,
        "last_5": last_5,
    }


def generate_match_preview(home, away, played, precomputed_matchup, elo_final=None):
    """Compose a 3-5 sentence preview using ALREADY-COMPUTED matchup values.

    precomputed_matchup must contain: p_home, p_draw, p_away, home_elo, away_elo,
    most_likely_score, p_most_likely. For knockout matches also: is_knockout,
    p_home_advances, p_away_advances.

    Never calls predict_matchup() internally — all probability/lambda values come
    from the caller's already-computed result so every display site shows identical
    numbers.
    """
    home_form = get_team_form(home, played, n=10, elo_final=elo_final)
    away_form = get_team_form(away, played, n=10, elo_final=elo_final)
    h2h = _get_h2h(home, away, played)
    m = precomputed_matchup

    def _form_sentence(team, form):
        n = form["n_matches"]
        if n == 0:
            return f"{team} have limited recent match history in the dataset."
        rs = form["result_string"]
        wins, draws, losses = rs.count("W"), rs.count("D"), rs.count("L")
        unbeaten_rate = wins + draws
        gf, ga = form["avg_goals_for"], form["avg_goals_against"]
        streak = form["current_streak"]
        if streak == unbeaten_rate and streak > 0:
            desc = f"on a {streak}-game unbeaten run"
        else:
            desc = f"{wins}W-{draws}D-{losses}L in their last {n}"
        return f"{team} arrive {desc}, averaging {gf:.1f} scored and {ga:.1f} conceded per match."

    sentences = [_form_sentence(home, home_form), _form_sentence(away, away_form)]

    if h2h["n_meetings"] > 0:
        n, hw, d, aw = h2h["n_meetings"], h2h["home_wins"], h2h["draws"], h2h["away_wins"]
        parts = []
        if hw: parts.append(f"{home} won {hw}")
        if d: parts.append(f"drawn {d}")
        if aw: parts.append(f"{away} won {aw}")
        sentences.append(f"In their last {n} meeting{'s' if n > 1 else ''}, {', '.join(parts)}.")
    else:
        sentences.append(f"No previous meetings between {home} and {away} are recorded in the dataset.")

    elo_diff = m["home_elo"] - m["away_elo"]
    if abs(elo_diff) < 50:
        elo_desc = f"teams closely matched on Elo ({m['home_elo']:.0f} vs {m['away_elo']:.0f})"
    elif elo_diff > 0:
        elo_desc = f"{home} holding a {elo_diff:.0f}-point Elo edge ({m['home_elo']:.0f} vs {m['away_elo']:.0f})"
    else:
        elo_desc = f"{away} holding a {abs(elo_diff):.0f}-point Elo edge ({m['away_elo']:.0f} vs {m['home_elo']:.0f})"
    sentences.append(
        f"The model gives {home} a {m['p_home']}% win probability, draw {m['p_draw']}%, "
        f"{away} {m['p_away']}%, with {elo_desc}; "
        f"most likely scoreline is {m['most_likely_score']} ({m['p_most_likely']}%)."
    )

    if m.get("is_knockout"):
        sentences.append(
            f"Factoring in extra time and a potential penalty shootout, {home} have a "
            f"{m['p_home_advances']}% chance of advancing versus {m['p_away_advances']}% for {away}."
        )

    return " ".join(sentences)


# ============================================================
# FULL PIPELINE — one call, returns everything a UI needs
# ============================================================
def run_pipeline(force_refresh=False):
    df = load_data(force_refresh=force_refresh)
    feat, elo_final, form_final = build_features(df)
    played = feat.dropna(subset=["home_score"]).copy()
    gbm = train_model(played)
    rho = fit_rho(played, gbm)
    bt = run_backtest(gbm, played, rho=rho)
    preds = predict_future(gbm, feat, rho=rho)
    teams = sorted(elo_final.keys())
    return dict(gbm=gbm, elo_final=elo_final, form_final=form_final, teams=teams,
                backtest=bt, predictions=preds, n_matches=len(played), rho=rho,
                played=played)
