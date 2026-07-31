# MBM player props engine (PRA v1)

Points + rebounds + assists for NBA and WNBA. No market inputs: the model reads box scores,
projects a player's total, models the spread around it, and turns that into a probability for
whatever line a book posts.

## Frozen 2026 holdout, opened once

| | |
|---|---|
| player-games | 16,553 |
| projection MAE | 6.548 |
| last-10 baseline MAE | 6.702 |
| log loss | 0.63223 |
| base rate | 0.68990 |
| **skill score** | **8.4%** |
| largest calibration miss | 0.019 |

Development window 2021-2025 read 7.8% on 81,016 walk-forward player-games. The board sits
between CS2 and NFL on the stack.

## What it does

**Projection.** A ridge on the player's own prior-games form: season average, last 5 and last
10 PRA, last 3 and last 10 minutes, their own recent volatility, rest, and whether they
started last time.

**Spread, not a constant.** A thirty-point scorer varies far more than an eight-point bench
player, so the spread is fitted against the projection rather than assumed. Without that the
tails are wrong for everyone.

**Calibration fitted out of sample.** The uncorrected model overstates the over by about 2.5
points across the middle of the range. That is not a location error and a shifted projection
does not fix it, so the correction is fitted on the probability scale using a held-out
trailing slice the projection never saw. Largest miss falls from 0.037 to 0.011.

## Two findings that shaped the build

**The model is worse on stars, not better.** Skill by rank within a team: 5.7% for the team's
top scorer, rising monotonically to 8.2% for the fourth option. A star's output has more
moving parts, drawing the hard defensive assignment, being game-planned against, sitting in
blowouts. A rotation player posts his number. So the board does not restrict to headliners.

**PRA errors correlate positively, both within and across teams.** Top two on the same team
r = +0.077; top one on each side r = +0.091. Shared pace, blowouts and overtime swamp the
usage tradeoff. Two overs never diversify, and a book pricing a two-leg PRA parlay by
multiplying is underpricing it.

## What it does not claim

Calibrated means it prints a fair price, not an edge. Bet at exactly the number and you break
even. Everything comes from finding a book offering better.

That matters more here than on the game boards, because props carry 6.5% to 9% hold against a
side's 4.5%. The edge required is roughly double, and prop lines already contain most of what
this model knows.

Where it is most likely to pay is stale lines: a player whose minutes just changed because a
teammate was ruled out, where the injury feed has moved and the prop has not.

## Operating

Runs 10am and 4pm Mountain. Every playable player on the slate is logged and graded; only the
eight per league where the projection is furthest from the player's own last-10 get pushed.
Players listed Out on the injury feed are dropped.

Playable means 10+ games this season and an 18-minute rolling average. Below that the line is
noise.

**Tripwire:** last-400 projection MAE against the last-10 baseline. If the model stops beating
"what this guy usually does", the board says so and stops being worth betting.

## Files

    daily_props.py       runner
    engine_props.py      features, fit, calibration, ladder
    board_format.py      shared line format and price floor
    notify.py            Discord and Telegram
    tg_delta.py          refresh gate
    spec.json            frozen settings
