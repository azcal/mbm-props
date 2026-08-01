# MBM player props engine (PRA v1)

Points + rebounds + assists for NBA and WNBA. No market inputs: the model reads box scores,
projects a player's total, models the spread around it, and turns that into a probability for
whatever line a book posts.

## Frozen 2026 holdout, opened once

| | |
|---|---|
| | NBA | WNBA |
|---|---|---|
| boardable player-games | 8,137 | 780 |
| **relative error** | **27.4%** | **28.9%** |
| projection MAE | 6.608 | 6.356 |
| last-10 baseline MAE | 6.944 | 6.400 |
| edge over baseline | +0.336 | +0.044 |
| largest calibration miss | 0.016 | 0.065 |
| median boardable per night | 44 | 12 |

**One fit, both leagues pooled.** Pooling helps each of them: NBA MAE 6.614 to 6.608, WNBA
6.387 to 6.356. WNBA alone has a fifth the history, so it borrows the shape of the
relationship from the larger league. A league indicator was tested and made both slightly
worse, which says the two leagues behave the same once minutes and form are accounted for.

**Read the WNBA numbers with caution.** Its edge over the naive baseline is +0.044 points
against the NBA's +0.336, and its calibration miss is four times as large on a 780-game
holdout. The board runs, and the tripwire watches it, but on current evidence the NBA side is
doing the work.

Relative error, |actual - projection| / projection, is the headline rather than MAE. Raw MAE
just tracks how big the number is: a 34-point scorer will always miss by more than a
12-point one without being harder to forecast. It is also why WNBA shows a lower MAE than
NBA while being the weaker board, since WNBA totals are smaller.

Skill score is deliberately not quoted as a headline. It moves with where the lines sit
relative to the base rate, so it flatters low-minutes players whose totals sit far from any
plausible line. Two filters that made skill look better made accuracy worse.

## What it does

**Projection.** A ridge on the player's own prior-games form: season average, last 5 and last
10 PRA, last 3 and last 10 minutes, their own recent volatility, rest, and whether they
started last time. Plus two team-level inputs:

**Teammates out** is the single largest addition, worth +0.156 points of absolute error at
t = 8.1. It measures what share of the team's usual production is missing tonight, and it is
the same input that carries two thirds of the NBA game board. It helps most where it should:
on heavily depleted rosters, MAE falls from 7.607 to 7.303, and on reinforced ones from 6.738
to 6.444. Historically the actives come from the box score; live they come from the injury
feed, since the box score does not exist yet.

**Opponent defence**, PRA conceded over the opponent's prior ten relative to the season mean,
adds a further 0.15%. Small but free.

**Spread, not a constant.** A thirty-point scorer varies far more than an eight-point bench
player, so the spread is fitted against the projection rather than assumed. Without that the
tails are wrong for everyone.

**Calibration fitted out of sample.** The uncorrected model overstates the over by about 2.5
points across the middle of the range. That is not a location error and a shifted projection
does not fix it, so the correction is fitted on the probability scale using a held-out
trailing slice the projection never saw. Largest miss falls from 0.037 to 0.011.

## What is boardable, and why

**Minutes, then minutes stability.** Both were measured on relative error and both separate
cleanly.

| filter | best quartile | worst quartile |
|---|---|---|
| rolling minutes | 25.0% | 43.0% |
| minutes dispersion (sd/mean) | 26.4% | 41.7% |

Applied together at 24 minutes and a 0.20 coefficient of variation, the board goes from 33.4%
relative error to 27.8% and from about 88 players a night to 44.

The dispersion filter is what catches a player flexing up and down to cover injuries. A
separate trend filter was tested and dropped: on its own it separates 32.1% from 36.1%, but
once minutes and dispersion are applied it adds 0.2 points while cutting the board by a
fifth. A role that is genuinely moving shows up as dispersion before it shows up as a clean
slope.

**What was tested and measured at zero:** home/away, 20 and 30 game averages, per-36 rate
form separated from volume, and modelling points, rebounds and assists separately before
summing. Each moved relative error by 0.01% or less. Gradient boosting instead of ridge was
worth 0.12% on the base features and is not worth the loss of a linear model you can read.

**Position, experience and shooting splits do nothing.** Position runs 31.0% to 34.4% with no
useful ordering. Years in the league has no ordering at all. Three-point rate matters slightly
because threes are lumpier, but it is small enough to ignore.

**A divergence filter was tested and removed.** Pushing only the players where the projection
was furthest from their own last-10 seemed like the NCAAF top-20 rule applied here. Measured,
skill by gap band came out flat at 7.0% to 9.2% with no ordering, so it was selecting
different forecasts rather than better ones. Everything playable is now pushed.

**An earlier claim in this file was wrong.** It said the model was worse on stars, 5.7% skill
against 8.2% for fourth options. That was skill score doing the thing described above. On
relative error, high-minutes players are far more predictable, and high minutes correlates
with being a primary option.

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

Runs 10am and 4pm Mountain. Every playable player on the slate is logged, graded and pushed.
Players listed Out on the injury feed are dropped.

Playable means 10+ games this season, a 24-minute rolling average, and minutes dispersion at
or under 0.20.

**Tripwire:** last-400 projection MAE against the last-10 baseline. If the model stops beating
"what this player usually does", the board says so and stops being worth betting. This matters
more for WNBA than NBA, because the WNBA margin over that baseline is already thin.

## Files

    daily_props.py       runner
    engine_props.py      features, fit, calibration, ladder
    board_format.py      shared line format and price floor
    notify.py            Discord and Telegram
    tg_delta.py          refresh gate
    spec.json            frozen settings
