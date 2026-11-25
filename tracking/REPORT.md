# SwishAI daily report

Updated 2026-11-25 02:16. This file is rewritten automatically by `python -m src.daily`.

## Where things stand

- **5,561 predictions graded** over 35 game days this season (1,293 made live before tip-off, the rest filled in afterwards from pre-game data).
- **Blend error this season: 4.40 points per game** (simple weighted average: 4.82; last season's locked test: 4.42).
- **Blend in use: weights + offset**, weights hybrid 0.54 / neural nets 0.40 / simulation 0.06; offset -0.10; bench / mid / star offsets +0.00 / +0.00 / +0.00.
- **222 predictions waiting** for games on 2026-11-25.

## Last graded night: 2026-11-24

69 players graded. Blend missed by **4.04** points on average; 87% of scores landed inside the 80% range.

**Top expected scorers**

| Player | Game | Predicted | 80% range | Actual | Miss |
| --- | --- | --- | --- | --- | --- |
| Jayson Tatum | BOS v WAS | 25.5 | 15-37 | 41 | +15.5 |
| Nikola Jokić | DEN v DAL | 23.4 | 14-35 | 18 | -5.5 |
| Jamal Murray | DEN v DAL | 23.1 | 14-35 | 20 | -3.3 |
| Cooper Flagg | DAL v DEN | 22.8 | 14-35 | 24 | +1.0 |
| Franz Wagner | ORL v DET | 21.6 | 12-33 | 20 | -1.8 |
| Kyrie Irving | DAL v DEN | 20.7 | 12-32 | 11 | -9.8 |
| Anthony Davis | WAS v BOS | 19.9 | 11-30 | 33 | +13.0 |
| Paul George | BOS v WAS | 17.5 | 9-28 | 28 | +10.4 |

**Biggest misses, and the main reason for each**

| Player | Predicted | Actual | Hybrid | Neural | Sim | Main reason |
| --- | --- | --- | --- | --- | --- | --- |
| Jayson Tatum | 25.5 | 41 | 25.4 | 25.8 | 25 | shot volume: 31 shots, expected 23 (+6.7 pts) |
| Anthony Davis | 20.0 | 33 | 19.9 | 20.1 | 20 | shot volume: 25 shots, expected 17 (+5.9 pts) |
| Aaron Gordon | 12.6 | 1 | 12.6 | 12.7 | 12 | shot volume: 4 shots, expected 11 (-6.4 pts) |
| Neemias Queta | 10.2 | 21 | 10.4 | 10.0 | 10 | shot volume: 12 shots, expected 8 (+6.3 pts) |
| Isaiah Joe | 10.2 | 21 | 9.9 | 10.6 | 10 | minutes: played 30, expected 19 (+5.7 pts) |
| Paul George | 17.6 | 28 | 17.5 | 17.7 | 18 | shooting: 1.40 pts per shot, usually 1.14 (+5.2 pts) |

## Which model is doing better or worse?

MAE = average points missed (lower is better). Bias = average of prediction minus actual. Beaten = how often the player scored more than predicted. The ML models aim for the typical (median) game, so Beaten near 50% is the sign of a fair model; a slightly negative Bias is normal, because rare huge nights pull the average up. 'vs blend' compares each model with the fixed blend on the same games; the verdict only says better or worse when the gap is bigger than its uncertainty (+/-).

**Season to date (5,561 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.821 | +0.82 | 40% | +0.423 | 0.052 | worse than the blend |
| Ridge regression | 4.491 | +0.14 | 45% | +0.093 | 0.021 | worse than the blend |
| Hybrid: linear + LightGBM | 4.400 | -0.38 | 48% | +0.002 | 0.007 | no real difference yet |
| Neural network ensemble | 4.403 | -0.47 | 48% | +0.006 | 0.009 | no real difference yet |
| Simulation median | 4.433 | -0.47 | 50% | +0.036 | 0.016 | worse than the blend |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.397 | -0.42 | 48% | +0.000 | 0.000 | the reference |

**Last 7 game days (1,110 games)**

| Model | MAE | Bias | Beaten | vs blend | +/- | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Simple weighted average (no ML) | 4.590 | +0.19 | 44% | +0.224 | 0.095 | worse than the blend |
| Ridge regression | 4.407 | -0.30 | 46% | +0.042 | 0.041 | worse than the blend |
| Hybrid: linear + LightGBM | 4.371 | -0.74 | 49% | +0.005 | 0.013 | no real difference yet |
| Neural network ensemble | 4.376 | -0.87 | 50% | +0.011 | 0.017 | no real difference yet |
| Simulation median | 4.363 | -0.90 | 52% | -0.002 | 0.033 | no real difference yet |
| Blend, fixed 0.5 / 0.4 / 0.1 | 4.365 | -0.81 | 50% | +0.000 | 0.000 | the reference |

**Live picks only (1,293 games predicted before tip-off):** official SwishAI number 4.482 MAE, fixed blend 4.476 MAE.

## Should the blend weights change?

Replaying 4,979 predictions day by day, 'weights + offset' beat the fixed weights by 0.0027 points per game (uncertainty +/-0.0027). The gain is bigger than its uncertainty, so it is switched on.

| policy | MAE | vs fixed | SE |
| --- | --- | --- | --- |
| Fixed weights 0.5/0.4/0.1 | 4.3713 | +0.0000 | 0.0000 |
| Adaptive weights | 4.3713 | -0.0000 | 0.0005 |
| Adaptive weights + one bias fix | 4.3685 | -0.0027 | 0.0027 |
| Adaptive weights + bias fix per tier | 4.3688 | -0.0024 | 0.0019 |
| Adaptive, NO shrinkage (for comparison) | 4.3699 | -0.0014 | 0.0054 |

Weights the adaptive method currently suggests: hybrid 0.54 / neural nets 0.40 / simulation 0.06, one offset -0.10, or per tier (bench / mid / star) -0.08 / -0.10 / +0.07. An option is only used if the table above shows it winning clearly.

## Why are we off? The error budget

Every miss is split into three parts that add up exactly: he played more or fewer **minutes** than expected, he took more or fewer **shots** per minute, or his shots **went in** more or less than usual.

| Part of the miss | Average size (points) |
| --- | --- |
| Minutes | 1.79 |
| Shot volume | 2.59 |
| Shooting (mostly luck) | 2.59 |

| If we knew... | MAE |
| --- | --- |
| only what we know now | 4.40 |
| + his exact minutes | 3.69 |
| + how many shots he took | 2.59 |

Reading: knowing exact minutes would cut the error from 4.40 to 3.69. Even knowing minutes AND shot count, 2.59 points of error remain: that is shooting luck, which no pre-game model can remove. So the realistic gains are in predicting minutes and role.

## Blind spots: groups where we lean the same way

A fair prediction is beaten half the time. 'Beaten' shows how often players in each group scored more than the models predicted (before any automatic correction). A group is flagged only when that is far from 50% beyond chance (z of 3 or more) AND the typical gap is at least 0.4 points. Lean = typical prediction minus actual (negative = too low). The last three columns show the typical miss from each part.

| Group | Games | MAE | Beaten | Lean | z | minutes | volume | shooting | Flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3+ days of rest | 931 | 3.93 | 44% | +0.51 | -3.8 | +0.14 | -0.26 | +0.00 | YES |
| Mid scorers (10-20 ppg) | 1746 | 5.06 | 47% | +0.45 | -2.7 | -0.24 | -0.08 | +0.38 |  |
| No teammates out | 820 | 4.36 | 45% | +0.51 | -2.7 | +0.06 | -0.25 | +0.01 |  |
| Returning after missing 3+ team games | 174 | 3.13 | 42% | +0.48 | -2.1 | +0.18 | -0.42 | +0.00 |  |
| Home games | 2771 | 4.38 | 48% | +0.25 | -2.0 | -0.08 | -0.17 | +0.00 |  |
| Opponent missing 15+ ppg | 2900 | 4.42 | 48% | +0.20 | -1.9 | -0.10 | -0.23 | +0.00 |  |
| Bench players (<10 ppg) | 3177 | 3.68 | 48% | +0.16 | -1.8 | +0.04 | -0.41 | +0.00 |  |
| Away games | 2790 | 4.42 | 48% | +0.17 | -1.8 | -0.05 | -0.44 | +0.23 |  |
| Veterans (33 or older) | 529 | 4.48 | 46% | +0.44 | -1.7 | -0.01 | -0.47 | +0.01 |  |
| Likely blowout (big strength gap) | 2236 | 4.38 | 49% | +0.20 | -1.4 | -0.04 | -0.41 | +0.22 |  |

## Did we guess the lineups right? (live picks)

- Players we expected to play: 1,474. Did not play: 181 (12%).
- Players who played but we did not expect: 95.
- Wrong lineup guesses hurt twice: that player's own prediction, and his teammates' (their 'teammates out' input was wrong).

## Actions and reasoning

**Done automatically this run**

- No change this run. The blend policy from the last review stays in place (see 'Should the blend weights change?' above).

**Recommended (waiting for your go-ahead)**

- **Do not retrain yet.** *Why:* only 5,561 new games since the last training (threshold 20,000). On 2026-11-25 a retrain on about 4,500 new games was tested fairly and gave no gain (they are only ~3% of the training data), so it was rolled back.
- **3+ days of rest**: predictions run too HIGH. Players beat our number 44% of the time over 931 games (50% is fair), typical gap 0.51 points, mostly through **volume**. Most off: nn model; least off: sim. With the automatic correction in use, that becomes 45%. Version 2 added 'player extra rest', which separates the team's days off from games he sat out. If the lean stays, the next idea is the reason for the absence from the injury report.
- **Biggest lever overall: minutes.** *Why:* perfect minutes knowledge would be worth 0.70 points of MAE, far more than any model swap (the models differ by about 0.01). Better injury, lineup and blowout information is where accuracy can still be won.

## Next predictions: 2026-11-25

222 players across 9 games. Top 25 shown; all are in `tracking/predictions.csv`.

Lineups assume: played his team's last game, and not listed as out on the injury report. Left out as injured: Aday Mara, Cameron Boozer, Jerami Grant, Nolan Traore, Norman Powell, OG Anunoby.

| Player | Game | Predicted | 80% range | Exp. minutes | Teammates out (ppg) |
| --- | --- | --- | --- | --- | --- |
| Shai Gilgeous-Alexander | OKC v MIN | 30.1 | 20-44 | 33 | 28 |
| Victor Wembanyama | SAS @ POR | 27.8 | 17-41 | 29 | 19 |
| Jalen Brunson | NYK v CLE | 26.2 | 16-38 | 34 | 27 |
| Giannis Antetokounmpo | MIA @ PHI | 25.3 | 15-37 | 31 | 16 |
| Devin Booker | PHX @ HOU | 23.5 | 14-36 | 32 | 8 |
| Ty Jerome | MEM v LAC | 23.4 | 13-35 | 29 | 29 |
| Tyrese Maxey | PHI v MIA | 23.0 | 14-34 | 35 | 7 |
| Donovan Mitchell | CLE @ NYK | 22.5 | 13-32 | 33 | 4 |
| Joel Embiid | PHI v MIA | 22.4 | 13-34 | 32 | 7 |
| Jaylen Brown | PHI v MIA | 22.2 | 13-32 | 33 | 7 |
| Deni Avdija | POR v SAS | 21.7 | 12-32 | 30 | 0 |
| Tyler Herro | MIL v BKN | 21.3 | 12-33 | 31 | 8 |
| Darius Acuff Jr. | SAC @ GSW | 21.2 | 13-34 | 32 | 1 |
| Karl-Anthony Towns | NYK v CLE | 21.1 | 13-32 | 32 | 27 |
| Julius Randle | BKN @ MIL | 20.8 | 12-32 | 32 | 19 |
| Brandon Miller | CHA v CHI | 20.6 | 11-32 | 31 | 14 |
| Kevin Durant | HOU v PHX | 20.2 | 11-31 | 31 | 0 |
| Darius Garland | LAC @ MEM | 19.4 | 10-31 | 32 | 7 |
| Michael Porter Jr. | BKN @ MIL | 19.3 | 10-30 | 32 | 19 |
| Bam Adebayo | MIA @ PHI | 19.1 | 10-29 | 32 | 16 |
| Naz Reid | CHA v CHI | 19.0 | 10-29 | 32 | 14 |
| Alperen Sengun | HOU v PHX | 18.8 | 10-29 | 29 | 0 |
| Damian Lillard | POR v SAS | 18.7 | 9-29 | 28 | 0 |
| Brandon Ingram | LAC @ MEM | 18.1 | 10-29 | 27 | 7 |
| Yaxel Lendeborg | GSW v SAC | 17.5 | 9-28 | 28 | 0 |
