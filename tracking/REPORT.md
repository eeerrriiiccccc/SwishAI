# SwishAI daily report

Updated 2026-11-21 13:26. This file is rewritten automatically by `python -m src.daily`.

## Where things stand

- **5,015 predictions graded** over 32 game days this season (794 made live before tip-off, the rest filled in afterwards from pre-game data).
- **Blend error this season: 4.39 points per game** (simple weighted average: 4.84; last season's locked test: 4.42).
- **Blend in use: weights + offset**, weights hybrid 0.61 / neural nets 0.34 / simulation 0.06; offset -0.11; bench / mid / star offsets +0.00 / +0.00 / +0.00.
- **107 predictions waiting** for games on 2026-11-21.

## Last graded night: 2026-11-21

20 players graded. Blend missed by **5.62** points on average; 85% of scores landed inside the 80% range.

**Top expected scorers**

| Player | Game | Predicted | 80% range | Actual | Miss |
| --- | --- | --- | --- | --- | --- |
| Tyrese Maxey | PHI v MIL | 24.5 | 16-37 | 17 | -7.5 |
| Joel Embiid | PHI v MIL | 24.5 | 15-37 | 26 | +1.6 |
| Tyler Herro | MIL v PHI | 17.7 | 9-29 | 50 | +32.2 |
| LeBron James | PHI v MIL | 17.1 | 10-27 | 21 | +3.8 |
| VJ Edgecombe | PHI v MIL | 15.6 | 8-25 | 16 | +0.3 |
| Jaime Jaquez Jr. | MIL v PHI | 13.2 | 6-23 | 9 | -4.3 |
| Anfernee Simons | PHI v MIL | 12.7 | 5-22 | 18 | +5.1 |
| Kyle Kuzma | MIL v PHI | 11.9 | 6-22 | 12 | -0.0 |

**Biggest misses, and the main reason for each**

| Player | Predicted | Actual | Hybrid | Neural | Sim | Main reason |
| --- | --- | --- | --- | --- | --- | --- |
| Tyler Herro | 17.8 | 50 | 18.0 | 17.5 | 18 | shooting: 1.74 pts per shot, usually 1.17 (+16.4 pts) |
| Brayden Burries | 10.0 | 26 | 10.3 | 9.4 | 11 | shot volume: 20 shots, expected 9 (+13.0 pts) |
| Dominick Barlow | 4.3 | 12 | 4.0 | 4.6 | 5 | shot volume: 6 shots, expected 4 (+4.9 pts) |
| Tyrese Maxey | 24.5 | 17 | 24.2 | 24.5 | 26 | shot volume: 15 shots, expected 21 (-7.4 pts) |
| Nate Ament | 7.6 | 2 | 8.0 | 7.1 | 8 | shot volume: 2 shots, expected 7 (-2.7 pts) |
| AJ Green | 3.7 | 9 | 3.7 | 3.8 | 3 | shooting: 3.00 pts per shot, usually 1.20 (+5.4 pts) |

## Which model is doing better or worse?

MAE = average points missed (lower is better). Bias = average of prediction minus actual. Beaten = how often the player scored more than predicted. The ML models aim for the typical (median) game, so Beaten near 50% is the sign of a fair model; a slightly negative Bias is normal, because rare huge nights pull the average up. 'vs blend' compares each model with the fixed blend on the same games; the verdict only says better or worse when the gap is bigger than its uncertainty (+/-).

**Season to date (5,015 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.842 | +0.89 | 40% | +0.450 | 0.055 | worse than the blend |
| Ridge regression | 4.493 | +0.19 | 44% | +0.100 | 0.022 | worse than the blend |
| Hybrid: linear + LightGBM | 4.394 | -0.33 | 48% | +0.001 | 0.008 | no real difference yet |
| Neural network ensemble | 4.400 | -0.42 | 48% | +0.007 | 0.010 | no real difference yet |
| Simulation median | 4.430 | -0.42 | 49% | +0.038 | 0.017 | worse than the blend |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.393 | -0.38 | 48% | +0.000 | 0.000 | the reference |

**Last 7 game days (1,040 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.648 | +0.15 | 45% | +0.221 | 0.098 | worse than the blend |
| Ridge regression | 4.460 | -0.32 | 47% | +0.033 | 0.043 | no real difference yet |
| Hybrid: linear + LightGBM | 4.426 | -0.79 | 51% | -0.001 | 0.014 | no real difference yet |
| Neural network ensemble | 4.448 | -0.91 | 51% | +0.021 | 0.018 | worse than the blend |
| Simulation median | 4.404 | -0.94 | 53% | -0.023 | 0.034 | no real difference yet |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.427 | -0.85 | 51% | +0.000 | 0.000 | the reference |

**Live picks only (794 games predicted before tip-off):** official SwishAI number 4.406 MAE, fixed blend 4.400 MAE.

## Should the blend weights change?

Replaying 4,433 predictions day by day, 'weights + offset' beat the fixed weights by 0.0037 points per game (uncertainty +/-0.0030). The gain is bigger than its uncertainty, so it is switched on.

| policy | MAE | vs fixed | SE |
| --- | --- | --- | --- |
| Fixed weights 0.5/0.4/0.1 | 4.3627 | +0.0000 | 0.0000 |
| Adaptive weights | 4.3625 | -0.0002 | 0.0005 |
| Adaptive weights + one bias fix | 4.3590 | -0.0037 | 0.0030 |
| Adaptive weights + bias fix per tier | 4.3590 | -0.0037 | 0.0021 |
| Adaptive, NO shrinkage (for comparison) | 4.3600 | -0.0026 | 0.0060 |

Weights the adaptive method currently suggests: hybrid 0.61 / neural nets 0.34 / simulation 0.06, one offset -0.11, or per tier (bench / mid / star) -0.07 / -0.17 / +0.08. An option is only used if the table above shows it winning clearly.

## Why are we off? The error budget

Every miss is split into three parts that add up exactly: he played more or fewer **minutes** than expected, he took more or fewer **shots** per minute, or his shots **went in** more or less than usual.

| Part of the miss | Average size (points) |
| --- | --- |
| Minutes | 1.80 |
| Shot volume | 2.60 |
| Shooting (mostly luck) | 2.58 |

| If we knew... | MAE |
| --- | --- |
| only what we know now | 4.39 |
| + his exact minutes | 3.69 |
| + how many shots he took | 2.58 |

Reading: knowing exact minutes would cut the error from 4.39 to 3.69. Even knowing minutes AND shot count, 2.58 points of error remain: that is shooting luck, which no pre-game model can remove. So the realistic gains are in predicting minutes and role.

## Blind spots: groups where we lean the same way

A fair prediction is beaten half the time. 'Beaten' shows how often players in each group scored more than the models predicted (before any automatic correction). A group is flagged only when that is far from 50% beyond chance (z of 3 or more) AND the typical gap is at least 0.4 points. Lean = typical prediction minus actual (negative = too low). The last three columns show the typical miss from each part.

| Group | Games | MAE | Beaten | Lean | z | minutes | volume | shooting | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3+ days of rest | 857 | 4.00 | 44% | +0.57 | -3.7 | +0.13 | -0.28 | +0.00 | YES |
| Mid scorers (10-20 ppg) | 1578 | 5.06 | 46% | +0.58 | -3.4 | -0.16 | -0.05 | +0.33 | YES |
| No teammates out | 770 | 4.40 | 46% | +0.45 | -2.4 | +0.07 | -0.28 | +0.00 |  |
| Home games | 2498 | 4.38 | 48% | +0.27 | -2.2 | -0.06 | -0.15 | +0.00 |  |
| Away games | 2517 | 4.40 | 48% | +0.22 | -2.0 | -0.05 | -0.43 | +0.21 |  |
| Opponent missing 15+ ppg | 2664 | 4.41 | 48% | +0.22 | -1.9 | -0.09 | -0.23 | +0.00 |  |
| Returning after missing 3+ team games | 148 | 3.13 | 43% | +0.50 | -1.8 | +0.15 | -0.38 | +0.00 |  |
| Veterans (33 or older) | 484 | 4.45 | 46% | +0.46 | -1.7 | +0.05 | -0.35 | +0.00 |  |
| Bench players (<10 ppg) | 2850 | 3.66 | 48% | +0.15 | -1.7 | +0.04 | -0.40 | +0.00 |  |
| Likely blowout (big strength gap) | 2013 | 4.39 | 48% | +0.24 | -1.5 | -0.02 | -0.37 | +0.20 |  |

## Did we guess the lineups right? (live picks)

- Players we expected to play: 910. Did not play: 116 (13%).
- Players who played but we did not expect: 48.
- Wrong lineup guesses hurt twice: that player's own prediction, and his teammates' (their 'teammates out' input was wrong).

## Actions and reasoning

**Done automatically this run**

- **blend policy**: weights_tier_offset -> weights_offset; weights 0.61/0.34/0.06 -> 0.61/0.34/0.06; offset +0.00 -> -0.11; bench/mid/star offsets -0.07/-0.19/+0.08 -> +0.00/+0.00/+0.00. *Why:* Replaying 4,433 predictions day by day, 'weights + offset' beat the fixed weights by 0.0037 points per game (uncertainty +/-0.0030). The gain is bigger than its uncertainty, so it is switched on.

**Recommended (waiting for your go-ahead)**

- **Retrain the models** (`python -m src.daily --retrain`). *Why:* 5,015 games have been played since the models were last trained. Retraining lets them learn this season's patterns and gives rookies their own player fingerprint. It takes a few minutes and replaces the saved model, so it waits for you.
- **Mid scorers (10-20 ppg)**: predictions run too HIGH. Players beat our number 46% of the time over 1578 games (50% is fair), typical gap 0.58 points, mostly through **shooting**. Most off: nn model; least off: sim. With the automatic correction in use, that becomes 47%. This season teams use deeper rotations, which costs mid scorers the most. Version 2 added rotation depth and season timing, which lowered the error but not the lean. The per-tier bias correction now works on it; it applies only part of the gap at first and more as games confirm it.
- **3+ days of rest**: predictions run too HIGH. Players beat our number 44% of the time over 857 games (50% is fair), typical gap 0.57 points, mostly through **volume**. Most off: nn model; least off: sim. With the automatic correction in use, that becomes 45%. Version 2 added 'player extra rest', which separates the team's days off from games he sat out. If the lean stays, the next idea is the reason for the absence from the injury report.
- **Biggest lever overall: minutes.** *Why:* perfect minutes knowledge would be worth 0.70 points of MAE, far more than any model swap (the models differ by about 0.01). Better injury, lineup and blowout information is where accuracy can still be won.

## Next predictions: 2026-11-21

107 players across 5 games. Top 25 shown; all are in `tracking/predictions.csv`.

Lineups assume: played his team's last game, and not listed as out on the injury report. Left out as injured: Cameron Boozer, Derrick White, Dylan Harper, Javon Small, Jayson Tatum, Jerami Grant, Jordan Clarkson, Mitchell Robinson, Trey Murphy III.

| Player | Game | Predicted | 80% range | Exp. minutes | Teammates out (ppg) |
| --- | --- | --- | --- | --- | --- |
| Victor Wembanyama | SAS v NOP | 29.0 | 17-41 | 31 | 22 |
| Jalen Brunson | NYK v POR | 24.9 | 15-36 | 34 | 14 |
| Darius Acuff Jr. | SAC v BOS | 22.8 | 14-33 | 36 | 11 |
| Cooper Flagg | DAL v LAC | 22.5 | 13-34 | 31 | 5 |
| Paul George | BOS @ SAC | 21.4 | 11-32 | 31 | 61 |
| Deni Avdija | POR @ NYK | 20.9 | 12-31 | 31 | 5 |
| Zion Williamson | NOP @ SAS | 20.9 | 13-32 | 33 | 38 |
| Kyrie Irving | DAL v LAC | 20.6 | 11-32 | 30 | 5 |
| Damian Lillard | POR @ NYK | 19.7 | 10-29 | 31 | 5 |
| Karl-Anthony Towns | NYK v POR | 19.4 | 11-29 | 33 | 14 |
| Darius Garland | LAC @ DAL | 18.8 | 10-31 | 30 | 3 |
| LaMelo Ball | MIN v MEM | 18.8 | 9-28 | 29 | 8 |
| Stephon Castle | SAS v NOP | 18.2 | 9-28 | 30 | 22 |
| Cedric Coward | MEM @ MIN | 18.2 | 9-28 | 31 | 78 |
| Payton Pritchard | BOS @ SAC | 17.8 | 9-28 | 30 | 61 |
| Jeremiah Fears | NOP @ SAS | 17.6 | 9-27 | 33 | 38 |
| Saddiq Bey | NOP @ SAS | 17.1 | 9-27 | 34 | 38 |
| Zach LaVine | SAC v BOS | 16.7 | 9-26 | 34 | 11 |
| OG Anunoby | NYK v POR | 16.4 | 8-26 | 34 | 14 |
| Ayo Dosunmu | MIN v MEM | 16.1 | 8-25 | 31 | 8 |
| Jaden McDaniels | MIN v MEM | 15.6 | 9-26 | 32 | 8 |
| Cam Spencer | MEM @ MIN | 14.2 | 6-23 | 30 | 78 |
| De'Aaron Fox | SAS v NOP | 13.9 | 7-23 | 28 | 22 |
| Devin Vassell | SAS v NOP | 13.7 | 6-23 | 30 | 22 |
| Jaylen Wells | MEM @ MIN | 13.5 | 6-22 | 29 | 78 |
