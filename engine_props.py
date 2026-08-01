"""
MBM player props engine -- points + rebounds + assists, NBA and WNBA.

No market inputs. The model reads box scores, builds a projection and a spread around it,
and turns those into a probability for any line a book posts.

Two design notes worth knowing.

**The spread is modelled, not assumed.** A thirty-point scorer varies far more than an
eight-point bench player, so a constant residual would price the tails wrong for everyone.
It is fitted on absolute residuals as a function of the projection and the player's own
recent volatility.

**Two corrections, both fitted out of sample.** The fit holds out its most recent quarter,
prices it with a model that never saw it, and learns both from those predictions. Fitting
either in-sample would just relearn the bias.

*Shrinkage slope.* Ridge pulls everything toward the middle, so low scorers are projected too
high and high scorers too low: measured at -0.5 on sub-12 projections and +0.9 on 30-plus. A
linear recalibration of the projection flattens that, and it matters because one point of PRA
is roughly four to five points of over/under probability.

*Probability level.* Even with the slope fixed, the raw normal overstates the over by a couple
of points across the middle. A two-parameter correction on the probability scale handles it.

Measured on 113,923 walk-forward NBA player-games: log loss 0.628 against a 0.690 base rate,
a 9.0% skill score, largest calibration miss 0.037 before correction.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

FEATS = ["std_pra", "l10_pra", "l5_pra", "l10_min", "l3_min", "l10_sd",
         "n_prior", "rest", "prev_start", "mates_out", "opp_def"]
# Playability. Relative error, |actual - projection| / projection, is the honest accuracy
# measure here: raw MAE just tracks how big the number is, so a 34-point scorer always looks
# worse than a 12-point one without being harder to forecast.
#
# Minutes is by far the strongest filter, running 43.0% relative error in the lowest quartile
# down to 25.0% in the highest, cleanly monotonic. Minutes *stability* is second: 26.4% for
# the steadiest quartile against 41.7% for the most erratic, which is the player flexing up
# and down to cover injuries. Applied together they take the board from 33.4% to 27.8%.
#
# Trend was tested and dropped. On its own it separates 32.1% from 36.1%, and once minutes
# and stability are applied it adds 0.2 points while cutting the board by a fifth. The
# variability filter already catches a player whose role is moving, because a role change
# shows up as dispersion before it shows up as a clean slope.
MIN_PRIOR = 10          # games this season before a player is boardable
MIN_MINUTES = 24.0      # rolling minutes floor
MAX_MIN_CV = 0.20       # minutes dispersion: sd over mean across the last 10


def build(P):
    """Per-player-game features, every one from prior games only."""
    P = P.copy()
    P["game_date"] = pd.to_datetime(P.game_date)
    P["pra"] = P.points + P.rebounds + P.assists
    P["mins"] = pd.to_numeric(P.minutes, errors="coerce")
    P = P.sort_values(["athlete_id", "game_date"]).reset_index(drop=True)
    g = P.groupby(["athlete_id", "season"])
    P["n_prior"] = g.cumcount()
    P["std_pra"] = g.pra.transform(lambda s: s.shift().expanding().mean())
    P["l10_pra"] = g.pra.transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    P["l5_pra"] = g.pra.transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    P["l10_min"] = g.mins.transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    P["l3_min"] = g.mins.transform(lambda s: s.shift().rolling(3, min_periods=2).mean())
    P["l10_sd"] = g.pra.transform(lambda s: s.shift().rolling(10, min_periods=4).std())
    P["prev_start"] = g.starter.transform(lambda s: s.shift()).fillna(0).astype(float)
    P["rest"] = g.game_date.transform(lambda s: (s - s.shift()).dt.days).clip(0, 7).fillna(3)
    P["min_sd"] = g.mins.transform(lambda s: s.shift().rolling(10, min_periods=6).std())
    P["min_cv"] = P.min_sd / P.l10_min.replace(0, np.nan)

    # Opponent defence: PRA conceded over their prior ten, relative to the season mean.
    opp = (P.groupby(["opponent_team_id", "season", "game_date"]).pra.sum()
           .reset_index().sort_values("game_date"))
    opp["allowed"] = (opp.groupby(["opponent_team_id", "season"]).pra
                      .transform(lambda s: s.shift().rolling(10, min_periods=4).mean()))
    P = P.merge(opp[["opponent_team_id", "season", "game_date", "allowed"]],
                on=["opponent_team_id", "season", "game_date"], how="left")
    P["opp_def"] = (P.allowed - P.groupby("season").allowed.transform("mean")).fillna(0.0)

    # Team baseline production, for the teammates-out feature. Prior games only.
    tm = (P.groupby(["team_id", "season", "game_date"]).pra.sum().reset_index()
          .sort_values("game_date"))
    tm["team_norm"] = (tm.groupby(["team_id", "season"]).pra
                       .transform(lambda s: s.shift().rolling(10, min_periods=4).mean()))
    P = P.merge(tm[["team_id", "season", "game_date", "team_norm"]],
                on=["team_id", "season", "game_date"], how="left")
    return P


def mates_out(P, actives=None):
    """How much of a team's usual production is missing tonight, as a share.

    This is the single biggest remaining feature, worth +0.156 points of absolute error at
    t = 8.1, and it is the same input that carries two thirds of the NBA game board. It uses
    tonight's roster, which is known at tipoff, against prior-game production. Nothing from
    tonight's result enters it.

    Two paths. Historically the actives come from who appears in the box score. Live they
    come from the injury feed, since the box score does not exist yet, so the runner passes
    an `actives` set of athlete ids and everyone else on the roster is treated as out.
    """
    P = P.copy()
    if actives is None:
        present = P.groupby(["team_id", "game_date"]).l10_pra.transform("sum")
    else:
        avail = P.athlete_id.astype(str).isin(actives)
        present = (P.l10_pra * avail).groupby([P.team_id, P.game_date]).transform("sum")
    P["mates_out"] = ((P.team_norm - present) / P.team_norm.replace(0, np.nan)).fillna(0.0)
    return P


def playable(D):
    """Enough history, enough minutes, and a role that is not moving."""
    return (D.n_prior >= MIN_PRIOR) & (D.l10_min >= MIN_MINUTES) & \
           (D.min_cv.notna()) & (D.min_cv <= MAX_MIN_CV) & \
           D[FEATS].notna().all(axis=1)


def _raw(sc, mean, sp, D):
    mu = mean.predict(sc.transform(D[FEATS].values))
    sd = np.clip(sp.predict(np.c_[mu, D["l10_sd"].values]) * 1.2533, 2.0, None)
    return mu, sd


def fit(D, tr, alpha=5.0, holdout=0.25, seed=7):
    """Projection, spread, a shrinkage correction and a probability correction.

    The last two are fitted on a held-out trailing slice the projection never saw."""
    idx = np.where(np.asarray(tr))[0]
    order = D.loc[idx].sort_values("game_date").index.values
    cut = int(len(order) * (1 - holdout))
    core, val = order[:cut], order[cut:]

    def _core(rows):
        X, y = D.loc[rows, FEATS].values, D.loc[rows, "pra"].values
        sc = StandardScaler().fit(X)
        mean = Ridge(alpha=alpha).fit(sc.transform(X), y)
        mu = mean.predict(sc.transform(X))
        sp = Ridge(alpha=alpha).fit(np.c_[mu, D.loc[rows, "l10_sd"].values], np.abs(y - mu))
        return sc, mean, sp

    # price the held-out slice with a model that never saw it, then learn both corrections
    sc0, m0, s0 = _core(core)
    mu_v, sd_v = _raw(sc0, m0, s0, D.loc[val])
    # 1. undo the shrinkage: regress the truth on the projection
    shrink = Ridge(alpha=1e-6).fit(mu_v.reshape(-1, 1), D.loc[val, "pra"].values)
    mu_v = shrink.predict(mu_v.reshape(-1, 1))
    # Sweep the correction fit in units of the player's own spread, not in fixed points.
    # A fixed +-6 sweep is a wide net around a 10-point bench player and a narrow one around
    # a 34-point starter, so on a tighter board it stops matching where lines actually sit.
    rng = np.random.default_rng(seed)
    off = rng.choice([-1.0, -0.6, -0.35, -0.15, 0.15, 0.35, 0.6, 1.0], len(val)) * sd_v
    L = np.round((mu_v + off) * 2) / 2
    L = np.where(L % 1 == 0, L + 0.5, L)
    p_raw = 1 - stats.norm.cdf((L - mu_v) / sd_v)
    y_over = (D.loc[val, "pra"].values > L).astype(int)
    lg = np.log(np.clip(p_raw, 1e-6, 1-1e-6) / np.clip(1 - p_raw, 1e-6, 1-1e-6))
    cal = LogisticRegression(C=1e6, max_iter=2000).fit(lg.reshape(-1, 1), y_over)

    # refit the projection on everything now both corrections are settled
    sc, mean, sp = _core(order)
    return {"sc": sc, "mean": mean, "sp": sp, "cal": cal, "shrink": shrink,
            "feats": FEATS}


def project(H, D):
    """Projected PRA, shrinkage-corrected, and the spread around it."""
    mu, sd = _raw(H["sc"], H["mean"], H["sp"], D)
    return H["shrink"].predict(mu.reshape(-1, 1)), sd


def median(H, mu, sd, grid=np.arange(-4, 4.01, 0.05)):
    """The point where the corrected probability crosses 0.5.

    PRA is right-skewed, floored at zero with a long tail, so the mean sits above the
    median. The median is the forecast: the number a player is as likely to beat as miss,
    which is what a posted line should be compared to.
    """
    mu = np.atleast_1d(np.asarray(mu, float)); sd = np.atleast_1d(np.asarray(sd, float))
    out = np.zeros(len(mu))
    for i in range(0, len(mu), 4000):
        sl = slice(i, min(i + 4000, len(mu)))
        cand = mu[sl][:, None] + grid[None, :]
        pp = np.column_stack([p_over(H, mu[sl], sd[sl], cand[:, j])
                              for j in range(len(grid))])
        out[sl] = cand[np.arange(len(mu[sl])), np.abs(pp - 0.5).argmin(axis=1)]
    return out


def p_over(H, mu, sd, line):
    """Corrected probability the player finishes above a posted line."""
    p = 1 - stats.norm.cdf((np.asarray(line, float) - mu) / sd)
    lg = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1 - p, 1e-6, 1-1e-6))
    return H["cal"].predict_proba(np.asarray(lg).reshape(-1, 1))[:, 1]


def ladder(H, mu_i, sd_i, step=2.0, span=2):
    """Half-point lines around the projection, with the corrected probability for each."""
    base = np.round(mu_i * 2) / 2
    lines = []
    for k in range(-span, span + 1):
        L = base + k * step
        lines.append(L + 0.5 if L % 1 == 0 else L)
    ps = p_over(H, np.full(len(lines), mu_i), np.full(len(lines), sd_i), np.array(lines))
    return list(zip(lines, [float(x) for x in ps]))
