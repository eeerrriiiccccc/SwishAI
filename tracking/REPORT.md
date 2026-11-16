# SwishAI daily report

Updated 2026-11-15 23:26. This file is rewritten automatically by `python -m src.daily`.

## Where things stand

- **4,173 predictions graded** over 26 game days this season (0 made live before tip-off, the rest filled in afterwards from pre-game data).
- **Blend error this season: 4.42 points per game** (simple weighted average: 4.90; last season's locked test: 4.42).
- **Blend in use: weights + tier + offset**, weights hybrid 0.54 / neural nets 0.40 / simulation 0.06; offset +0.00; bench / mid / star offsets -0.03 / -0.30 / +0.05.
- **190 predictions waiting** for games on 2026-11-16.

## Last graded night: 2026-11-15

198 players graded. Blend missed by **5.05** points on average; 77% of scores landed inside the 80% range.

**Top expected scorers**

| Player | Game | Predicted | 80% range | Actual | Miss |
| --- | --- | --- | --- | --- | --- |
| Luka Dončić | LAL v PHX | 35.3 | 24-50 | 43 | +7.7 |
| Shai Gilgeous-Alexander | OKC v SAC | 29.9 | 18-43 | 37 | +7.1 |
| Victor Wembanyama | SAS v UTA | 27.7 | 14-38 | 26 | -1.7 |
| Tyrese Maxey | PHI v IND | 25.8 | 16-38 | 27 | +1.2 |
| Jalen Brunson | NYK v TOR | 25.8 | 15-36 | 28 | +2.2 |
| Darius Acuff Jr. | SAC v OKC | 25.8 | 16-38 | 26 | +0.2 |
| Jayson Tatum | BOS v DAL | 25.2 | 15-36 | 32 | +6.8 |
| Devin Booker | PHX v LAL | 25.1 | 15-36 | 17 | -8.1 |

**Biggest misses, and the main reason for each**

| Player | Predicted | Actual | Hybrid | Neural | Sim | Main reason |
| --- | --- | --- | --- | --- | --- | --- |
| Kasparas Jakučionis | 4.3 | 25 | 4.2 | 4.2 | 5 | shot volume: 17 shots, expected 4 (+12.4 pts) |
| Mo Bamba | 6.8 | 26 | 7.1 | 6.4 | 7 | shooting: 1.82 pts per shot, usually 1.20 (+8.9 pts) |
| Trae Young | 18.8 | 0 | 18.9 | 18.6 | 19 | shot volume: 5 shots, expected 16 (-8.6 pts) |
| Collin Gillespie | 8.6 | 26 | 8.8 | 8.5 | 8 | shot volume: 17 shots, expected 8 (+8.6 pts) |
| Tyler Herro | 21.6 | 5 | 22.0 | 21.3 | 21 | shooting: 0.37 pts per shot, usually 1.19 (-10.9 pts) |
| Landry Shamet | 7.8 | 23 | 7.8 | 7.7 | 8 | shooting: 1.73 pts per shot, usually 1.16 (+7.6 pts) |

## Which model is doing better or worse?

MAE = average points missed (lower is better). Bias = average of prediction minus actual. Beaten = how often the player scored more than predicted. The ML models aim for the typical (median) game, so Beaten near 50% is the sign of a fair model; a slightly negative Bias is normal, because rare huge nights pull the average up. 'vs blend' compares each model with the fixed blend on the same games; the verdict only says better or worse when the gap is bigger than its uncertainty (+/-).

**Season to date (4,173 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.901 | +1.02 | 39% | +0.486 | 0.063 | worse than the blend |
| Ridge regression | 4.525 | +0.29 | 44% | +0.110 | 0.025 | worse than the blend |
| Hybrid: linear + LightGBM | 4.417 | -0.25 | 48% | +0.001 | 0.009 | no real difference yet |
| Neural network ensemble | 4.419 | -0.33 | 48% | +0.004 | 0.011 | no real difference yet |
| Simulation median | 4.466 | -0.33 | 49% | +0.051 | 0.019 | worse than the blend |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.415 | -0.29 | 47% | +0.000 | 0.000 | the reference |

**Last 7 game days (1,122 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.797 | +0.26 | 44% | +0.344 | 0.104 | worse than the blend |
| Ridge regression | 4.512 | -0.18 | 48% | +0.058 | 0.045 | worse than the blend |
| Hybrid: linear + LightGBM | 4.451 | -0.66 | 51% | -0.003 | 0.015 | no real difference yet |
| Neural network ensemble | 4.457 | -0.85 | 51% | +0.004 | 0.020 | no real difference yet |
| Simulation median | 4.504 | -0.80 | 52% | +0.051 | 0.033 | worse than the blend |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.454 | -0.75 | 51% | +0.000 | 0.000 | the reference |

## Should the blend weights change?

Replaying 3,591 predictions day by day, 'weights + tier + offset' beat the fixed weights by 0.0055 points per game (uncertainty +/-0.0023). The gain is bigger than its uncertainty, so it is switched on.

| policy | MAE | vs fixed | SE |
| --- | --- | --- | --- |
| Fixed weights 0.5/0.4/0.1 | 4.3820 | +0.0000 | 0.0000 |
| Adaptive weights | 4.3816 | -0.0004 | 0.0006 |
| Adaptive weights + one bias fix | 4.3770 | -0.0050 | 0.0035 |
| Adaptive weights + bias fix per tier | 4.3765 | -0.0055 | 0.0023 |
| Adaptive, NO shrinkage (for comparison) | 4.3775 | -0.0045 | 0.0072 |

Weights the adaptive method currently suggests: hybrid 0.54 / neural nets 0.40 / simulation 0.06, one offset -0.15, or per tier (bench / mid / star) -0.03 / -0.30 / +0.05. An option is only used if the table above shows it winning clearly.

## Why are we off? The error budget

Every miss is split into three parts that add up exactly: he played more or fewer **minutes** than expected, he took more or fewer **shots** per minute, or his shots **went in** more or less than usual.

| Part of the miss | Average size (points) |
| --- | --- |
| Minutes | 1.82 |
| Shot volume | 2.61 |
| Shooting (mostly luck) | 2.57 |

| If we knew... | MAE |
| --- | --- |
| only what we know now | 4.42 |
| + his exact minutes | 3.70 |
| + how many shots he took | 2.57 |

Reading: knowing exact minutes would cut the error from 4.42 to 3.70. Even knowing minutes AND shot count, 2.57 points of error remain: that is shooting luck, which no pre-game model can remove. So the realistic gains are in predicting minutes and role.

## Blind spots: groups where we lean the same way

A fair prediction is beaten half the time. 'Beaten' shows how often players in each group scored more than the models predicted (before any automatic correction). A group is flagged only when that is far from 50% beyond chance (z of 3 or more) AND the typical gap is at least 0.4 points. Lean = typical prediction minus actual (negative = too low). The last three columns show the typical miss from each part.

| Group | Games | MAE | Beaten | Lean | z | minutes | volume | shooting | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mid scorers (10-20 ppg) | 1314 | 5.01 | 44% | +0.78 | -4.2 | -0.08 | -0.03 | +0.42 | YES |
| 3+ days of rest | 789 | 4.08 | 44% | +0.51 | -3.2 | +0.13 | -0.32 | +0.00 | YES |
| Veterans (33 or older) | 406 | 4.32 | 44% | +0.65 | -2.5 | +0.14 | -0.30 | +0.01 |  |
| Away games | 2091 | 4.42 | 47% | +0.27 | -2.4 | -0.04 | -0.48 | +0.32 |  |
| No teammates out | 678 | 4.34 | 46% | +0.40 | -2.2 | +0.08 | -0.36 | +0.06 |  |
| Home games | 2082 | 4.41 | 48% | +0.33 | -2.2 | -0.01 | -0.15 | +0.00 |  |
| Returning after missing 3+ team games | 126 | 3.28 | 40% | +0.77 | -2.1 | +0.20 | -0.33 | +0.00 |  |
| Opponent missing 15+ ppg | 2235 | 4.43 | 48% | +0.20 | -1.6 | -0.09 | -0.20 | +0.00 |  |
| Likely blowout (big strength gap) | 1692 | 4.39 | 48% | +0.25 | -1.6 | +0.00 | -0.42 | +0.28 |  |
| Bench players (<10 ppg) | 2355 | 3.69 | 49% | +0.12 | -1.1 | +0.03 | -0.43 | +0.00 |  |

## Actions and reasoning

**Done automatically this run**

- No change this run. The blend policy from the last review stays in place (see 'Should the blend weights change?' above).

**Recommended (waiting for your go-ahead)**

- **Do not retrain yet.** *Why:* only 4,173 new games since the last training (threshold 5,000). Retraining on little new data changes the model without real gain.
- **Mid scorers (10-20 ppg)**: predictions run too HIGH. Players beat our number 44% of the time over 1314 games (50% is fair), typical gap 0.78 points, mostly through **shooting**. Most off: nn model; least off: sim. With the automatic correction in use, that becomes 45%. This season teams use deeper rotations, which costs mid scorers the most. Version 2 added rotation depth and season timing, which lowered the error but not the lean. The per-tier bias correction now works on it; it applies only part of the gap at first and more as games confirm it.
- **3+ days of rest**: predictions run too HIGH. Players beat our number 44% of the time over 789 games (50% is fair), typical gap 0.51 points, mostly through **volume**. Most off: nn model; least off: sim. With the automatic correction in use, that becomes 45%. Version 2 added 'player extra rest', which separates the team's days off from games he sat out. If the lean stays, the next idea is the reason for the absence from the injury report.
- **Biggest lever overall: minutes.** *Why:* perfect minutes knowledge would be worth 0.71 points of MAE, far more than any model swap (the models differ by about 0.01). Better injury, lineup and blowout information is where accuracy can still be won.

## Next predictions: 2026-11-16

190 players across 8 games. Top 25 shown; all are in `tracking/predictions.csv`.

Lineups assume: played his team's last game, and not listed as out on the injury report. Left out as injured: Bennedict Mathurin, Brandon Williams, Bryce McGowens, Caleb Wilson, Chris Youngblood, Dejounte Murray, Evan Mobley, Fred VanVleet, GG Jackson, Jaylen Clark, Marcus Smart, Stephen Curry, Tre Johnson, Ty Jerome.

| Player | Game | Predicted | 80% range | Exp. minutes | Teammates out (ppg) |
| --- | --- | --- | --- | --- | --- |
| Giannis Antetokounmpo | MIA v HOU | 26.1 | 15-37 | 32 | 7 |
| Devin Booker | PHX @ DEN | 25.6 | 15-36 | 35 | 19 |
| Donovan Mitchell | CLE v CHI | 24.2 | 14-36 | 32 | 16 |
| Jamal Murray | DEN v PHX | 23.9 | 14-34 | 37 | 12 |
| Tyrese Maxey | PHI @ BKN | 23.7 | 15-36 | 33 | 17 |
| Jayson Tatum | BOS v ORL | 23.7 | 13-35 | 32 | 6 |
| Nikola Jokić | DEN v PHX | 23.6 | 14-34 | 36 | 12 |
| Deni Avdija | POR @ MIN | 23.4 | 13-33 | 33 | 12 |
| Anthony Davis | WAS v MEM | 22.4 | 13-32 | 33 | 10 |
| Yaxel Lendeborg | GSW @ NOP | 22.0 | 12-33 | 31 | 38 |
| Kevin Durant | HOU @ MIA | 21.6 | 13-33 | 34 | 23 |
| Jaylen Brown | PHI @ BKN | 21.6 | 13-33 | 32 | 17 |
| Alperen Sengun | HOU @ MIA | 21.5 | 13-32 | 34 | 23 |
| Zion Williamson | NOP v GSW | 21.1 | 13-32 | 33 | 50 |
| Joel Embiid | PHI @ BKN | 20.8 | 11-31 | 30 | 17 |
| Norman Powell | CHI @ CLE | 20.7 | 11-31 | 30 | 26 |
| Damian Lillard | POR @ MIN | 20.5 | 11-30 | 31 | 12 |
| Franz Wagner | ORL @ BOS | 20.0 | 11-32 | 31 | 4 |
| Michael Porter Jr. | BKN v PHI | 19.2 | 10-30 | 29 | 0 |
| Bam Adebayo | MIA v HOU | 18.8 | 10-28 | 34 | 7 |
| LaMelo Ball | MIN v POR | 18.5 | 9-29 | 30 | 13 |
| James Harden | CLE v CHI | 18.2 | 10-29 | 31 | 16 |
| Julius Randle | BKN v PHI | 18.2 | 10-28 | 29 | 0 |
| Saddiq Bey | NOP v GSW | 17.9 | 10-28 | 33 | 50 |
| Brandin Podziemski | GSW @ NOP | 17.8 | 9-28 | 31 | 38 |
