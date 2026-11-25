# Sequential Bayesian Football Prediction

A prediction pipeline that looks for an edge on Premier League matches.

1. **Team ratings.** A Bayesian model, fitted with MCMC (Markov chain Monte Carlo), gives every team an attack and a defence rating. It also estimates one home-advantage term for the league. Ratings start from last season's results and are updated after every gameweek. They learn from a blend of goals and shots on target (see [The model](#the-model)), because shots are a less noisy measure of how well a team played.
2. **Match probabilities.** Each fixture's ratings become expected goals (xG) for both sides. A Dixon-Coles Poisson model turns those into scoreline probabilities, then into home/draw/away and over/under 2.5 probabilities.
3. **Bet sizing.** The over/under probabilities are compared with bookmaker odds, and fractional Kelly sizes a stake whenever the expected value is positive.

Everything (teams, results, fixtures, odds, ratings, predictions and orders) is stored in one SQLite file, `football.db`.

## Performance

Every prediction is made only from results played before that match, so no future data leaks in. The model is judged on log loss, where lower means better probabilities. The real benchmark is Bet365's odds with the bookmaker's margin removed, because beating the market is what a betting edge requires. Guessing a third each way scores 1.099 for home/draw/away, and a coin flip scores 0.693 for over/under.

| Test | Home/draw/away: model vs Bet365 | Over/under 2.5: model vs Bet365 |
|---|---|---|
| 2024/25 backtest, 380 matches (run 4) | 0.974 vs 0.971 | 0.677 vs 0.677 |
| 2025/26 backtest, 380 matches (run 8) | 1.025 vs 1.019 | 0.698 vs 0.689 |
| 2026/27 live, rounds 1–11, 110 matches (goals-only model) | 1.026 (no odds for rounds 3–11 yet) | 0.696 |

These backtests use the current shot weight of 0.75. The goals-only model (runs 1 and 2) scored 0.983 and 1.035 on home/draw/away and 0.692 and 0.710 on over/under. Adding shots closed half to two-thirds of the gap to Bet365, and it ties Bet365 on 2024/25 over/under. Bet365 is still more accurate everywhere else, and over/under in 2025/26 is still slightly worse than a coin flip. Weights from 0.25 to 1.0 are compared in runs 3–12: 0.75 to 0.9 score about the same, and shots alone (1.0) are worse.

Simulated betting results swing between profit and loss depending on the season, market and shot weight, all within what luck alone produces over about 350 bets. So nothing here is evidence of an edge. Re-run `python backtest/backtest.py --report <run id>` for the full tables.

## Setup

You need Python 3.9 or later. Install the dependencies:

```
pip install numpy pandas scipy
```

`football.db` is committed to the repo, so it works straight after a clone. SQLite comes with Python, so there's nothing else to install.

## Weekly workflow

Run these commands after each gameweek. The examples assume round 12 has just been played and round 13 is next. Seasons are 4-digit codes: `2627` means 2026/27.

```
# 1. Pull the latest results and fixtures (safe to re-run)
python fetch.py --season 2627

# 2. Pull shots on target for those results from ESPN (safe to re-run)
python fetch.py --season 2627 --shots

# 3. Add round 12's results to the ratings (reads round 11's ratings, writes round 12's)
python w_bhm.py --season 2627 --round 12

# 4. Predict round 13 using the round 12 ratings
python poisson.py --season 2627 --round 13

# 5. Add the bookmaker's over/under 2.5 odds for round 13
python fetch.py --season 2627 --odds odds_r13.csv

# 6. Size the bets
python kelly_sizing.py --season 2627 --round 13 --bankroll 1000 --fraction 0.25
```

Run step 2 before step 3. A match without shots data is rated on goals alone, and the round stays that way unless you re-run `w_bhm.py` for it and every round after it. Step 2 prints how many played matches still have no shots. ESPN's scoreboard is an unofficial, keyless source, so it may change without notice. It has matched football-data.co.uk's shots-on-target counts on 751 of 760 team counts in 2025/26.

Step 5's odds file needs these columns. Team names can be the canonical names or a known alias such as `Man Utd` or `Spurs`.

```
HomeTeam,AwayTeam,Odds_Over_25,Odds_Under_25
Arsenal,Chelsea,1.776,2.211
```

Without odds, step 6 places no bets; it just tells you which matches have no odds.

To predict a single match with the latest ratings:

```
python poisson.py --season 2627 --match Arsenal "Man City"
```

## Starting a new season

```
# 1. Load the new season's fixtures. Promoted teams are added on this first fetch.
python fetch.py --season 2728

# 2. Fit last season's results and save them as round 0 of the new season
python y_bhm.py --train 2627 --season 2728
```

Step 2 drops the relegated teams. Each promoted team starts with the relegated teams' average rating and a wider variance (`PROMOTED_VAR` in `y_bhm.py`), because the model has no top-flight data on them yet. From there, follow the weekly workflow starting with round 1.

**If a team shows up twice:** on the first fetch of a season, any team name the database doesn't recognise is added as a new team. If a feed spells an existing team differently, you'll get a duplicate. Fix it by deleting the new row from `teams` and adding the spelling to `team_aliases`.

To import an older season (football-data.co.uk CSV format: `Date, HomeTeam, AwayTeam, FTHG, FTAG`, plus `HST, AST` for shots on target if the file has them):

```
python fetch.py --season 2526 --history data/season-2526.csv
```

To add market odds for a season (Bet365 pre-match, plus the best price across about 17 bookmakers), use `Matches.csv` from [Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data-2000-2025). It republishes football-data.co.uk's odds. A row is only stored if its score matches the result already in the database.

```
python fetch.py --season 2526 --market-odds Matches.csv
```

## Backtesting

```
python backtest/backtest.py --season 2425 2526      # replay seasons (about 3 minutes each)
python backtest/backtest.py --list                  # saved runs
python backtest/backtest.py --report 2 --min-edge 0.05 --kelly 0.1
python backtest/backtest.py --report 2 --export     # CSVs for viz_1x2.py and viz_ou.py
python backtest/backtest.py --season 2526 --shot-weight 0   # goals-only ratings (default: SHOT_WEIGHT in db.py)
```

A backtest runs the live pipeline on a past season. It fits the preseason ratings on the season before, then steps through the season, predicting each match with the ratings of the time before adding its result. Steps follow match dates, so a postponed match is predicted and learned from when it was actually played. Each run saves its predictions, so runs with different settings (`--rho`, `--seed`, `--train`, `--shot-weight`) can be compared later. Each run's shot weight is saved in its note.

The report scores the model against Bet365's odds and simulates betting on the outcome with the highest edge in each market, at Bet365's price and at the best price across bookmakers. It gives flat-stake profit and a fractional Kelly bankroll.

Charts: `python backtest/viz_1x2.py --csv backtest/backtest_run2_1x2.csv` and `python backtest/viz_ou.py --csv backtest/backtest_run2_ou2_5.csv`. Both need `pip install matplotlib`, and `viz_ou.py` opens a window after saving its PNG.

## Project layout

| File | Purpose |
|---|---|
| `fetch.py` | Loads results and fixtures from fixturedownload.com, shots on target from ESPN, plus odds CSVs and historical season CSVs. Re-running only updates what changed. |
| `y_bhm.py` | Pre-season fit on past season(s), with an attack-tier mixture prior. Handles relegation and promotion. Saves round 0. |
| `w_bhm.py` | Weekly update: last round's ratings become the prior, and this round's results are the new evidence. |
| `poisson.py` | Dixon-Coles scoreline grid giving home/draw/away and over/under 2.5 probabilities. Saves predictions. |
| `kelly_sizing.py` | Fractional Kelly stakes for over/under 2.5 bets with positive expected value. Saves orders. |
| `db.py` | Database schema, the shared read/write functions, and the goals-and-shots blend (`SHOT_WEIGHT`). |
| `backtest/` | `backtest.py` (walk-forward backtest against market odds) and the chart scripts. |
| `data/` | The raw 2023–26 season CSVs the database was first built from. |

## The model

For a match between home team *h* and away team *a*:

```
home goals ~ Poisson(exp(intercept + home_adv + att[h] + def[a]))
away goals ~ Poisson(exp(intercept + att[a] + def[h]))
```

- **The evidence is a blend, not raw goals.** For rating updates, each side's "goals" are `(1 - w) × goals + w × 0.328 × shots on target`, with `w = 0.75` (`SHOT_WEIGHT` in `db.py`). 0.328 is the number of goals per shot on target over 2023/24–2025/26. A match with no shots data uses real goals. Predictions and backtest scoring always use real goals.
- A higher `att` means a team scores more. A higher `def` means it **concedes** more.
- The `intercept` is fixed at 0.2231 (that is, ln 1.25).
- After every MCMC sweep, attack and defence ratings are each recentred to sum to zero across the league's 20 teams.
- **Weekly updates:** each team's posterior is summarised as a Gaussian (median and variance). It becomes next week's prior, with a small amount of variance added (`time_decay = 0.005`, capped at 0.05) so ratings can follow changes in form.
- **Predictions:** a Dixon-Coles correction (`rho = -0.1285`) adjusts the probabilities of the low scorelines 0-0, 1-0, 0-1 and 1-1. The scoreline grid stops at 5 goals per side, so probabilities sum to about 99%.

## Database

| Table | Contents |
|---|---|
| `teams`, `team_aliases` | Team IDs and alternative spellings |
| `matches` | One row per fixture. Goals and shots on target (`home_sot`, `away_sot`) are empty until the match is played. The over/under odds columns hold the odds you enter for betting. |
| `rating_runs`, `ratings` | Ratings for each season and round. Round 0 is the preseason fit; round *N* includes round *N*'s results. |
| `odds` | Market odds per match and source: `bet365` and `best` (highest price across bookmakers), for home/draw/away and over/under 2.5. Every match 2023–26 and 2026/27 rounds 1–2. |
| `predictions` | The latest prediction for each match, with the rating round it came from |
| `orders` | Kelly stakes from `kelly_sizing.py` |
| `backtest_runs`, `backtest_predictions` | Each backtest's settings and its prediction for every match |

You can browse the data with any SQLite client, or from the command line:

```
sqlite3 football.db "SELECT * FROM ratings WHERE season='2627' AND round=11 ORDER BY att_mean DESC"
```

## Known limitations

- **Postponed matches in the weekly workflow.** A postponed match stays in its original round. `w_bhm.py` warns about it, but its result is skipped once that round has been processed. The backtest handles this by date.
- **Pre-match odds, not closing odds.** The `odds` table holds Bet365's pre-match prices. Closing odds are the sharper benchmark, and they aren't stored.
- **No reset between seasons.** Ratings carried over from last season are not shrunk toward 0 and their variances are not widened before round 1, so early-season ratings may be over-confident.
- **The intercept is fixed** and doesn't learn from data, so a league-wide change in scoring rate isn't picked up.
- **The shot weight is fixed.** 0.75 was picked from one backtest run per weight. A model that treats shots and goals as separate evidence could learn the weight itself.
- **Current-season shots come from an unofficial ESPN endpoint.** If it breaks, `fetch.py --shots` fails and new rounds are rated on goals only.
