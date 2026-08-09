"""
MBM player props daily runner -- points + rebounds + assists, NBA and WNBA.

Boards every player with enough history and minutes on tonight's slate, prices a ladder of
lines around the projection, and pushes only the ones where the model is furthest from the
player's own recent form. That last filter is the same idea as the NCAAF top-20 rule: the
edge, if there is one, lives where the model and the naive read disagree.

Players listed Out are dropped. No market inputs anywhere.

Usage: python daily_props.py [--date YYYY-MM-DD]
"""
import argparse, io, json, os, sys, time, datetime as dt
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_props as E
import board_format as BF
import espn_roster as R
import notify, tg_delta

# Shared files must be at least this new. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here. Raise it only for a change this
# runner depends on.
#   2026-08-02  tg_delta's optional "match" field, which this board uses to carry the
#               forecast, because its ladder prices barely move (see the push block).
SHARED_MIN = "2026-08-02"
for _mod in (BF, notify, tg_delta, R):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

DATA = os.path.join(HERE, "data")
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
H = {"User-Agent": "Mozilla/5.0"}          # raw.githubusercontent, fine with a UA
# ESPN 403s ANY custom User-Agent and only serves the requests default. Measured
# 2026-08-05 on the WNBA scoreboard: no header 200, "Mozilla/5.0" 403, an honest
# identifier with contact details 403, a full browser header set 403. mbm-wnba has
# always worked because it never set one. Do not "fix" this by adding a UA.
ESPN_H = {}

LEAGUES = {
    "NBA": dict(repo="hoopR-nba-data/main/nba", espn="basketball/nba",
                seasons=range(2021, 2028)),
    "WNBA": dict(repo="wehoop-wnba-data/main/wnba", espn="basketball/wnba",
                 seasons=range(2021, 2028)),
}
LADDER_STEP = SPEC.get("ladder_step", 2.0)
# Printed on every board header so a row says what it is forecasting. Without it "forecast
# 36" and "over 32.5" name no market at all, which is fine for one board and not fine the
# moment a second one exists. Kept in the spec so adding a market is a config change.
MARKET_TAG = SPEC.get("market_tag", "PTS+REB+AST")

# How recently a player must have logged minutes to be boardable.
#
# Without this, `latest` hands back each player's most recent state across every loaded
# season, so anyone whose final game was years ago still boards on any night their old team
# plays. Measured on the 2026-07-31 board: eleven of forty-one players never appeared in a
# box score, including three who retired before 2025. Neither existing filter catches them.
# The injury feed does not list a retired player as Out, and playable() cannot help because
# n_prior counts games within a season and theirs was a complete one.
#
# It is keyed on the last game with minutes rather than the last appearance on a roster,
# so a player who dresses without playing is dropped too. Twenty-one days survives a normal
# injury absence and the NBA All-Star break, and is far short of an offseason.
RECENT_DAYS = SPEC.get("recent_days", 21)

# Retired 2026-08-07. Forecasts are still produced, logged and graded every run; nothing
# reaches Discord and the dashboard shows this under Retired.
#
# The projection is genuinely better than naive, and that is not the same as bettable.
# Measured on the 2026 holdout, 8,137 NBA player-games: MAE 6.608 against a last-ten
# baseline of 6.944, a gain of 0.336 points. Against a typical standard deviation of 8.3
# that is about 1.6 percentage points of probability edge at the line. Prop markets hold
# 6.5% to 9%, needing roughly 3.2 to 4.5 points to break even, so the board is short by a
# factor of two to three. WNBA is worse: +0.052 points of MAE, about 0.26 points of
# probability, effectively nothing.
#
# And that 1.6 is measured against a naive last-ten average rather than against a book. A
# book is not using a last-ten average, so the real gap is smaller and possibly negative.
# Nothing in this stack carries market prices, so it cannot be measured either way.
#
# Set to False to publish again if a source of market prices ever makes that answerable.
RETIRED = True


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
    """Teams playing today, from the ESPN scoreboard.

    Reports why it came back empty. This used to swallow every exception and return {},
    which printed "no games" whether the league was out of season, ESPN had rate-limited
    us, or the response was not JSON. Those need different responses and looked identical.
    """
    url = (f"https://site.api.espn.com/apis/site/v2/sports/"
           f"{LEAGUES[lg]['espn']}/scoreboard")
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=ESPN_H,
                                params={"dates": day.strftime("%Y%m%d")},
                                timeout=45)
        except Exception as ex:
            print(f"{lg}: scoreboard unreachable ({type(ex).__name__}), attempt {attempt+1}")
            time.sleep(3)
            continue
        if resp.status_code == 200:
            try:
                r = resp.json()
                break
            except Exception:
                print(f"{lg}: scoreboard returned non-JSON, {resp.text[:80]!r}")
                return {}
        print(f"{lg}: scoreboard HTTP {resp.status_code}, attempt {attempt+1}")
        time.sleep(3)
    else:
        print(f"{lg}: scoreboard failed three times, treating as no slate")
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


def rosters_and_outs(lg, day, team_ids=None):
    """Tonight's rosters and who is unavailable, from the roster endpoint.

    Replaces the standalone /injuries call. Roster membership matters here for the same
    reason it does on the game boards: `latest.team_id` reads off a player's most recent
    box score, so a player traded this week is still filed under her old team until she
    appears in one for the new one.
    """
    base = f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[lg]['espn']}"
    # Only the teams on tonight's slate. The full sweep was one call per team, thirty for
    # NBA, which is a lot of requests against a host that starts refusing, for rosters
    # nothing on the board will ever read.
    ros = R.fetch(base, only=team_ids)
    outs = R.out_ids(ros)
    if ros["by_ab"]:
        R.log_statuses(ros, LOGS, day, lg)
    return ros, outs


def grade_from_espn(log):
    """Fill in PRA totals the hoopR and wehoop mirrors have not published yet.

    Same lag the game boards hit. Measured 2026-08-06: wehoop's latest completed WNBA game
    was 2026-08-01 while ESPN had every game through 2026-08-05 with full box scores. Four
    days of props sat ungraded, which stalls the rolling MAE and the tripwire.

    ESPN's summary endpoint returns per-athlete stat lines with a label row, so PRA is
    read by looking up PTS, REB and AST by name rather than by position. It only touches
    rows the mirrors left blank and walks back at most two weeks.
    """
    if "actual" not in log.columns or log.empty:
        return log
    missing = log[log.actual.isna()]
    if missing.empty:
        return log
    cutoff = str(pd.Timestamp.utcnow().date() - pd.Timedelta(days=14))
    missing = missing[missing.date.astype(str) >= cutoff]
    if missing.empty:
        return log

    found = {}
    for lg, grp in missing.groupby("league"):
        base = f"https://site.api.espn.com/apis/site/v2/sports/{LEAGUES[lg]['espn']}"
        for gid in sorted(set(grp.game_id.astype(str))):
            try:
                j = requests.get(f"{base}/summary", headers=ESPN_H,
                                 params={"event": gid}, timeout=30).json()
            except Exception:
                continue
            for blk in j.get("boxscore", {}).get("players", []):
                for st in blk.get("statistics", []):
                    labels = [str(x).upper() for x in (st.get("labels") or [])]
                    try:
                        ip, ir, ia = (labels.index("PTS"), labels.index("REB"),
                                      labels.index("AST"))
                    except ValueError:
                        continue
                    for a in st.get("athletes", []):
                        if a.get("didNotPlay"):
                            continue
                        aid = str((a.get("athlete") or {}).get("id", ""))
                        vals = a.get("stats") or []
                        if not aid or len(vals) <= max(ip, ir, ia):
                            continue
                        try:
                            # Keyed on game AND athlete. Keying on athlete alone would
                            # write one night's line onto every ungraded row that player
                            # has in the window, which is most of them mid-season.
                            found[(gid, aid)] = (float(vals[ip]) + float(vals[ir])
                                                 + float(vals[ia]))
                        except (TypeError, ValueError):
                            continue
            time.sleep(0.3)                      # this is one call per game, so be polite
    if not found:
        return log
    key = list(zip(log.game_id.astype(str), log.athlete_id.astype(str)))
    fill = pd.Series([found.get(k) for k in key], index=log.index, dtype="float64")
    n = int((log.actual.isna() & fill.notna()).sum())
    if n:
        log["actual"] = log.actual.fillna(fill)
        print(f"graded {n} prop(s) from ESPN that the mirrors have not published yet")
    return log


def run(day_str):
    day = dt.datetime.strptime(day_str, "%Y-%m-%d").date()
    cur = day.year + (1 if day.month >= 10 else 0)
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str, "athlete_id": str}) \
        if os.path.exists(pred_path) else pd.DataFrame()

    # One fit across both leagues. Pooling helps each: NBA MAE 6.614 -> 6.608 and WNBA
    # 6.387 -> 6.356. WNBA alone has a fifth the data and its edge over the naive baseline
    # is thin, so it borrows the shape of the relationship from the larger league. A league
    # indicator was tested and made both slightly worse, so the leagues really do behave the
    # same once minutes and form are accounted for.
    push, skipped, all_rows, console = {}, 0, [], []
    frames, raw = {}, {}
    for lg in LEAGUES:
        download(lg, cur)
        P = load(lg)
        raw[lg] = P          # kept for grading; re-reading every parquet twice was wasteful
        if P.empty:
            print(f"{lg}: no player data")
            continue
        D = E.mates_out(E.build(P))       # historical actives from the box score
        D = D[D.mins > 0].reset_index(drop=True)
        D["lg"] = lg
        frames[lg] = D
    if not frames:
        print("no player data for any league")
        return
    pool = pd.concat(frames.values(), ignore_index=True)
    hist = pool[E.playable(pool)].reset_index(drop=True)
    if len(hist) < 3000:
        print(f"only {len(hist)} usable rows pooled, not enough to fit")
        return
    heads = E.fit(hist, np.ones(len(hist), bool))
    print(f"pooled fit on {len(hist):,} player-games "
          f"({', '.join(f'{k} {int((hist.lg == k).sum()):,}' for k in frames)})")

    for lg, D in frames.items():

        teams = slate(lg, day)
        if not teams:
            print(f"{lg}: no games on {day}")
            continue
        ros, outs = rosters_and_outs(lg, day, set(teams))
        # each player's most recent state, for the teams playing today
        latest = (D.sort_values("game_date").groupby("athlete_id").tail(1)
                  .assign(team_id=lambda x: x.team_id.astype(str)))
        # Applied here, above both `tonight` and `roster`, so it also keeps inactive players
        # out of the actives set. Leaving them in there inflates `present` inside mates_out
        # and understates how depleted a team is, which distorts the projection of every
        # real player alongside them.
        # Team membership from the roster, overriding what the last box score says. Without
        # this a player traded in is still filed under her old team, so she is missing from
        # her new team's board and her new team's mates_out, and counted for her old one.
        # `teams` is keyed by ESPN team id, which is what by_id carries.
        if ros["by_id"]:
            aid2tid = {aid: tid for tid, ids in ros["by_id"].items() for aid in ids}
            mapped = latest.athlete_id.astype(str).map(aid2tid)
            moved = int((mapped.notna() & (mapped != latest.team_id)).sum())
            if moved:
                print(f"{lg}: {moved} players re-filed to their current roster team")
            latest = latest.assign(team_id=mapped.fillna(latest.team_id))

        cutoff = pd.Timestamp(day) - pd.Timedelta(days=RECENT_DAYS)
        stale = latest.team_id.isin(teams) & (latest.game_date < cutoff)
        if stale.any():
            names = list(latest.loc[stale, "athlete_display_name"].astype(str))
            print(f"{lg}: {int(stale.sum())} on tonight's teams have not logged minutes in "
                  f"{RECENT_DAYS} days, dropped: {', '.join(names[:8])}"
                  + (" ..." if len(names) > 8 else ""))
        latest = latest[latest.game_date >= cutoff]
        roster = latest[latest.team_id.isin(teams)]
        if roster.empty:
            print(f"{lg}: nobody rostered on tonight's teams")
            continue
        # Recompute teammates-out from tonight's actives. The historical version reads the
        # box score, which does not exist yet, so the injury feed stands in: everyone on the
        # roster who is not listed Out counts as available.
        #
        # Summed over the WHOLE roster, not the boardable subset. Historically `present`
        # comes from everyone who appeared in the box score, so restricting it to playable
        # players counts every bench player as missing and pushes the feature far outside
        # the range it was fitted on. Measured on 2026-08-01 WNBA: +0.441 the subset way
        # against a training 95th percentile of +0.179, and -0.088 this way against a
        # training mean of +0.003. playable() is applied after, so the boarded rows are the
        # same ones as before, carrying a feature value that is in distribution.
        actives = set(roster.athlete_id.astype(str)) - set(outs)
        pool = E.mates_out(roster.assign(game_date=pd.Timestamp(day)), actives)
        tonight = pool[~pool.athlete_id.astype(str).isin(outs)]
        tonight = tonight[E.playable(tonight)].copy()
        if tonight.empty:
            print(f"{lg}: no boardable players tonight")
            continue
        mu, sd = E.project(heads, tonight)
        med = E.median(heads, mu, sd)          # unfloored, used to centre the ladder
        tonight["proj"] = mu                     # the mean, kept for the log
        tonight["sd"] = sd
        tonight["median"] = np.floor(med)        # the forecast, what gets displayed
        tonight["med_raw"] = med                 # centres the ladder; flooring here would
                                                 # pull every rung down by up to a point
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
            lad = E.ladder(heads, r.proj, r.sd, step=LADDER_STEP, centre=r.med_raw)
            # The lowest rung that clears the price floor, which is the easiest line the
            # board actually publishes. Grading belongs here rather than on `grade_line`,
            # a line 2.5 above the forecast that exists only so the board is gradeable and
            # is never shown to anyone.
            pub = [(L, p) for L, p in lad
                   if BF.min_odds(p) >= BF.PRICE_FLOOR and p >= 0.20]
            line_lo, p_lo = (pub[0] if pub else (float("nan"), float("nan")))
            console.append(f"  {str(r.athlete_display_name)[:22]:<22} {r.matchup:<12} "
                           f"{MARKET_TAG} forecast {r.median:4.0f}  (mean {r.proj:.1f}, "
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
                             "gap": round(float(r.gap), 2),
                             "line_lo": float(line_lo), "p_lo": round(float(p_lo), 4),
                             "boarded": 0 if RETIRED else 1,
                             "model": "v1-retired" if RETIRED else "v1"})

        # Every playable player is pushed. A divergence filter was tested and dropped:
        # skill by gap band came out flat at 7.0% to 9.2% with no ordering, so selecting on
        # it was choosing different forecasts rather than better ones. The playability
        # filter in the engine does the real work.
        for r in tonight.sort_values("proj", ascending=False).itertuples():
            lad = E.ladder(heads, r.proj, r.sd, step=LADDER_STEP, centre=r.med_raw)
            best = max(lad, key=lambda x: x[1] if x[1] < 0.75 else 0)
            if BF.min_odds(best[1]) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            body = [f"{day.isoformat()}  {str(r.athlete_display_name)[:22]}  "
                    f"{r.matchup}  {MARKET_TAG} forecast {r.median:.0f}"]
            for L, p in lad:
                if BF.min_odds(p) >= BF.PRICE_FLOOR and p >= 0.20:
                    body.append(f"    over {L:.1f}   {p:>5.1%}  "
                                f"find {BF.price(p)} or better")
            # The gate's price check is nearly blind here. The ladder is built around the
            # projection, so its probabilities are close to invariant: across tonight's
            # board a forecast range of 16 to 36 PRA produced min_odds from 1.47 to 1.60,
            # against a 0.02 tolerance. A teammate being ruled out between the 10am and 4pm
            # runs moves the forecast several points and shifts every line, while the
            # prices barely budge. So the forecast itself goes in the "match" field, which
            # the gate compares and reports verbatim.
            push[f"{r.game_id}|{r.athlete_id}"] = {"line": "\n".join(body),
                                                   "min_odds": BF.min_odds(best[1]),
                                                   "match": f"forecast {r.median:.0f}"}

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
        # Not astype(str) on its own: athlete_id reads back from CSV as float64 whenever
        # the column has a blank, so the stored row keys on "2998938.0" and the new row on
        # "2998938", the old row is never dropped, and each rerun leaves a duplicate.
        _nk = lambda c: (c.astype(str).str.strip()
                         .str.replace(r"\.0$", "", regex=True))
        key = lambda d: _nk(d.date) + "|" + _nk(d.athlete_id)
        log = pd.concat([log[~key(log).isin(set(key(b)))] if len(log) else log, b],
                        ignore_index=True)

    # grade: join yesterday's rows to the box scores that have since landed
    alerts = []
    if len(log):
        actual = {}
        for lg, P in raw.items():
            if P.empty:
                continue
            P = P.assign(pra=P.points + P.rebounds + P.assists,
                         d=pd.to_datetime(P.game_date).dt.date.astype(str))
            actual.update(zip(P.d + "|" + P.athlete_id.astype(str),
                              P.pra.astype(float)))
        log["actual"] = (log.date.astype(str) + "|" + log.athlete_id.astype(str)).map(actual)
        log = grade_from_espn(log)
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

    if RETIRED:
        # Retired 2026-08-07, still forecast and graded, never published. See the note on
        # RETIRED above for the arithmetic. Alerts still fire: if the projection stops
        # beating the naive baseline that is worth knowing even for a board nobody bets.
        print(f"\nRETIRED board: {len(push)} row(s) forecast and logged, nothing sent.")
        if alerts:
            notify.send("MBM Props (retired)",
                        ["TRIPWIRE: " + ", ".join(alerts)
                         + " no longer beating the baseline."
                         + " This board is not published; flagged for the record."],
                        None, notify.RED)
        return

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
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
