"""
MBM player props daily runner -- points + rebounds + assists, NBA and WNBA.

Boards every player with enough history and minutes on tonight's slate, prices a ladder of
lines around the projection, and pushes only the ones where the model is furthest from the
player's own recent form. That last filter is the same idea as the NCAAF top-20 rule: the
edge, if there is one, lives where the model and the naive read disagree.

Players listed Out are dropped. No market inputs anywhere.

Usage: python daily_props.py [--date YYYY-MM-DD]
"""
import argparse, io, json, os, sys, datetime as dt
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_props as E
import board_format as BF
import notify, tg_delta

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
H = {"User-Agent": "Mozilla/5.0"}

LEAGUES = {
    "NBA": dict(repo="hoopR-nba-data/main/nba", espn="basketball/nba",
                seasons=range(2021, 2028)),
    "WNBA": dict(repo="wehoop-wnba-data/main/wnba", espn="basketball/wnba",
                 seasons=range(2021, 2028)),
}
PUSH_PER_LEAGUE = SPEC.get("push_per_league", 8)
LADDER_STEP = SPEC.get("ladder_step", 2.0)


def download(lg, cur):
    os.makedirs(DATA, exist_ok=True)
    base = f"https://raw.githubusercontent.com/sportsdataverse/{LEAGUES[lg]['repo']}"
    for y in LEAGUES[lg]["seasons"]:
        fn = os.path.join(DATA, f"{lg.lower()}_player_{y}.parquet")
        if os.path.exists(fn) and y < cur:
            continue
        try:
            r = requests.get(f"{base}/player_box/parquet/player_box_{y}.parquet",
                             headers=H, timeout=300)
            if r.status_code == 200:
                open(fn, "wb").write(r.content)
        except Exception:
            pass


def load(lg):
    frames = []
    for y in LEAGUES[lg]["seasons"]:
        fn = os.path.join(DATA, f"{lg.lower()}_player_{y}.parquet")
        if os.path.exists(fn):
            try:
                frames.append(pd.read_parquet(fn))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    P = pd.concat(frames, ignore_index=True)
    return P[(P.did_not_play != True) & P.points.notna() & P.minutes.notna()]


def slate(lg, day):
    """Teams playing today, from the ESPN scoreboard."""
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/"
                         f"{LEAGUES[lg]['espn']}/scoreboard",
                         headers=H, params={"dates": day.strftime("%Y%m%d")}, timeout=45).json()
    except Exception:
        return {}
    out = {}
    for ev in r.get("events", []):
        for c in ev.get("competitions", []):
            cs = c.get("competitors", [])
            if len(cs) != 2:
                continue
            for t in cs:
                opp = [x for x in cs if x is not t][0]
                out[str(t["team"]["id"])] = {
                    "abbr": t["team"].get("abbreviation", "?"),
                    "opp": opp["team"].get("abbreviation", "?"),
                    "home": t.get("homeAway") == "home",
                    "game_id": str(ev.get("id"))}
    return out


def out_ids(lg):
    """Athlete ids listed Out on the ESPN injury feed."""
    try:
        r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/"
                         f"{LEAGUES[lg]['espn']}/injuries", headers=H, timeout=45).json()
    except Exception:
        return set()
    bad = set()
    for team in r.get("injuries", []):
        for inj in team.get("injuries", []):
            if str(inj.get("status", "")).lower() in ("out", "suspension"):
                a = inj.get("athlete") or {}
                if a.get("id"):
                    bad.add(str(a["id"]))
    return bad


def run(day_str):
    day = dt.datetime.strptime(day_str, "%Y-%m-%d").date()
    cur = day.year + (1 if day.month >= 10 else 0)
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str, "athlete_id": str}) \
        if os.path.exists(pred_path) else pd.DataFrame()

    push, skipped, all_rows, console = {}, 0, [], []
    for lg in LEAGUES:
        download(lg, cur)
        P = load(lg)
        if P.empty:
            print(f"{lg}: no player data")
            continue
        D = E.build(P)
        D = D[D.mins > 0].reset_index(drop=True)
        hist = D[E.playable(D)].reset_index(drop=True)
        if len(hist) < 3000:
            print(f"{lg}: only {len(hist)} usable rows, not enough to fit")
            continue
        heads = E.fit(hist, np.ones(len(hist), bool))

        teams = slate(lg, day)
        if not teams:
            print(f"{lg}: no games on {day}")
            continue
        outs = out_ids(lg)
        # each player's most recent state, for the teams playing today
        latest = (D.sort_values("game_date").groupby("athlete_id").tail(1)
                  .assign(team_id=lambda x: x.team_id.astype(str)))
        tonight = latest[latest.team_id.isin(teams) & ~latest.athlete_id.astype(str).isin(outs)]
        tonight = tonight[E.playable(tonight)].copy()
        if tonight.empty:
            print(f"{lg}: no boardable players tonight")
            continue
        mu, sd = E.project(heads, tonight)
        med = E.median(heads, mu, sd)
        tonight["proj"] = mu                     # the mean, kept for the log
        tonight["sd"] = sd
        tonight["median"] = np.floor(med)        # the forecast, what gets displayed
        # a fixed line so the board is gradeable without knowing what a book posted,
        # the same trick the spread head uses with its -3.5
        tonight["grade_line"] = tonight["median"] + 2.5
        tonight["p_grade"] = E.p_over(heads, mu, sd, tonight.grade_line.values)
        tonight["gap"] = np.abs(tonight.proj - tonight.l10_pra)
        tonight["league"] = lg
        tonight["matchup"] = [f"{teams[t]['abbr']} vs {teams[t]['opp']}"
                              for t in tonight.team_id]
        tonight["game_id"] = [teams[t]["game_id"] for t in tonight.team_id]

        console.append(f"\n--- {lg} projections, {len(tonight)} players ---")
        for r in tonight.sort_values("gap", ascending=False).itertuples():
            lad = E.ladder(heads, r.proj, r.sd, step=LADDER_STEP)
            console.append(f"  {str(r.athlete_display_name)[:22]:<22} {r.matchup:<12} "
                           f"forecast {r.median:4.0f}  (mean {r.proj:.1f}, "
                           f"last10 {r.l10_pra:.1f})")
            console.append("      " + "   ".join(
                f"o{L:.1f} {p:.0%}" for L, p in lad))
            all_rows.append({"date": day.isoformat(), "league": lg,
                             "game_id": r.game_id, "athlete_id": str(r.athlete_id),
                             "player": r.athlete_display_name, "matchup": r.matchup,
                             "proj": round(float(r.proj), 2), "sd": round(float(r.sd), 2),
                             "median": float(r.median), "l10": round(float(r.l10_pra), 2),
                             "grade_line": float(r.grade_line),
                             "p_grade": round(float(r.p_grade), 4),
                             "gap": round(float(r.gap), 2), "model": "v1"})

        # push only where the model is furthest from the player's own recent form
        top = tonight.sort_values("gap", ascending=False).head(PUSH_PER_LEAGUE)
        for r in top.itertuples():
            lad = E.ladder(heads, r.proj, r.sd, step=LADDER_STEP)
            best = max(lad, key=lambda x: x[1] if x[1] < 0.75 else 0)
            if BF.min_odds(best[1]) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            body = [f"{day.isoformat()}  {str(r.athlete_display_name)[:22]}  "
                    f"{r.matchup}  forecast {r.median:.0f}"]
            for L, p in lad:
                if BF.min_odds(p) >= BF.PRICE_FLOOR and p >= 0.20:
                    body.append(f"    over {L:.1f}   {p:>5.1%}  "
                                f"find {BF.price(p)} or better")
            push[f"{r.game_id}|{r.athlete_id}"] = {"line": "\n".join(body),
                                                   "min_odds": BF.min_odds(best[1])}

    for ln in console:
        print(ln)
    print(f"\n=== MBM Props board ===")
    for v in push.values():
        print(v["line"])
    if skipped:
        print(f"({skipped} player{'s' if skipped != 1 else ''} skipped, "
              f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)})")

    if all_rows:
        b = pd.DataFrame(all_rows)
        key = lambda d: d.date.astype(str) + "|" + d.athlete_id.astype(str)
        log = pd.concat([log[~key(log).isin(set(key(b)))] if len(log) else log, b],
                        ignore_index=True)

    # grade: join yesterday's rows to the box scores that have since landed
    alerts = []
    if len(log):
        actual = {}
        for lg in LEAGUES:
            P = load(lg)
            if P.empty:
                continue
            P = P.assign(pra=P.points + P.rebounds + P.assists,
                         d=pd.to_datetime(P.game_date).dt.date.astype(str))
            for r in P.itertuples():
                actual[f"{r.d}|{r.athlete_id}"] = float(r.pra)
        log["actual"] = (log.date.astype(str) + "|" + log.athlete_id.astype(str)).map(actual)
        log.to_csv(pred_path, index=False)
        done = log[log.actual.notna()]
        if len(done) >= 50:
            mae = float(np.mean(np.abs(done.actual - done.proj)))
            base = float(np.mean(np.abs(done.actual - done.l10)))
            print(f"\nrolling grade: n={len(done)}  model MAE {mae:.3f}  "
                  f"last-10 baseline {base:.3f}  edge {base-mae:+.3f}")
            w = done.tail(400)
            if len(w) >= 400:
                m2 = float(np.mean(np.abs(w.actual - w.proj)))
                b2 = float(np.mean(np.abs(w.actual - w.l10)))
                ok = m2 < b2
                print(f"last-400 vs baseline: {b2-m2:+.3f}  "
                      f"[{'OK' if ok else 'TRIPWIRE: bench review'}]")
                if not ok:
                    alerts.append("Props PRA")

    mode, lines = tg_delta.gate(LOGS, day.strftime("%Y%m%d"), push)
    if lines or alerts:
        body = list(lines) if lines else ["no changes since the last push"]
        foot = (f"{skipped} skipped, price floor {BF.PRICE_FLOOR:.2f} / "
                f"{BF.american(BF.PRICE_FLOOR)}") if skipped else None
        title = "MBM Props" + (" · changes since the last push"
                               if mode == "delta" and lines else "")
        colour = notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " no longer beating the baseline, stop betting this board"]
            colour = notify.RED
        notify.send(title, body, foot, colour)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
