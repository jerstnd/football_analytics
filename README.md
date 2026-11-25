# Football Analytics

Premier League match predictions and over/under 2.5 bet sizing.

1. **Team ratings.** A Bayesian model, fitted with MCMC (Markov chain Monte Carlo), gives every team an attack and a defence rating. It also estimates one home-advantage term for the league. Ratings start from last season's results and are updated after every gameweek.
2. **Match probabilities.** Each fixture's ratings become expected goals for both sides. A Dixon-Coles Poisson model turns those into scoreline probabilities, then into home/draw/away and over/under 2.5 probabilities.
3. **Bet sizing.** The over/under probabilities are compared with bookmaker odds, and fractional Kelly sizes a stake whenever the expected value is positive.

Everything (teams, results, fixtures, odds, ratings, predictions and orders) is stored in one SQLite file, `football.db`.

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

# 2. Add round 12's results to the ratings (reads round 11's ratings, writes round 12's)
python w_bhm.py --season 2627 --round 12

# 3. Predict round 13 using the round 12 ratings
python poisson.py --season 2627 --round 13

# 4. Add the bookmaker's over/under 2.5 odds for round 13
python fetch.py --season 2627 --odds odds_r13.csv

# 5. Size the bets
python kelly_sizing.py --season 2627 --round 13 --bankroll 1000 --fraction 0.25
```

Step 4's odds file needs these columns. Team names can be the canonical names or a known alias such as `Man Utd` or `Spurs`.

```
HomeTeam,AwayTeam,Odds_Over_25,Odds_Under_25
Arsenal,Chelsea,1.776,2.211
```

Without odds, step 5 places no bets; it just tells you which matches have no odds.

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

To import an older season (football-data.co.uk CSV format: `Date, HomeTeam, AwayTeam, FTHG, FTAG`):

```
python fetch.py --season 2526 --history data/season-2526.csv
```

## Project layout

| File | Purpose |
|---|---|
| `fetch.py` | Loads results and fixtures from fixturedownload.com, plus odds CSVs and historical season CSVs. Re-running only updates what changed. |
| `y_bhm.py` | Pre-season fit on past season(s), with an attack-tier mixture prior. Handles relegation and promotion. Saves round 0. |
| `w_bhm.py` | Weekly update: last round's ratings become the prior, and this round's results are the new evidence. |
| `poisson.py` | Dixon-Coles scoreline grid giving home/draw/away and over/under 2.5 probabilities. Saves predictions. |
| `kelly_sizing.py` | Fractional Kelly stakes for over/under 2.5 bets with positive expected value. Saves orders. |
| `db.py` | Database schema and the shared read/write functions. |
| `backtest/` | Backtests against 2025/26 (see [Known limitations](#known-limitations)). |
| `data/` | Raw historical season CSVs, plus legacy files the backtests still use. |

## The model

For a match between home team *h* and away team *a*:

```
home goals ~ Poisson(exp(intercept + home_adv + att[h] + def[a]))
away goals ~ Poisson(exp(intercept + att[a] + def[h]))
```

- A higher `att` means a team scores more. A higher `def` means it **concedes** more.
- The `intercept` is fixed at 0.2231 (that is, ln 1.25).
- After every MCMC sweep, attack and defence ratings are each recentred to sum to zero across the league's 20 teams.
- **Weekly updates:** each team's posterior is summarised as a Gaussian (median and variance). It becomes next week's prior, with a small amount of variance added (`time_decay = 0.005`, capped at 0.05) so ratings can follow changes in form.
- **Predictions:** a Dixon-Coles correction (`rho = -0.1285`) adjusts the probabilities of the low scorelines 0-0, 1-0, 0-1 and 1-1. The scoreline grid stops at 5 goals per side, so probabilities sum to about 99%.

## Database

| Table | Contents |
|---|---|
| `teams`, `team_aliases` | Team IDs and alternative spellings |
| `matches` | One row per fixture. Goals are empty until the match is played, and odds are empty until imported. |
| `rating_runs`, `ratings` | Ratings for each season and round. Round 0 is the preseason fit; round *N* includes round *N*'s results. |
| `predictions` | The latest prediction for each match, with the rating round it came from |
| `orders` | Kelly stakes from `kelly_sizing.py` |

You can browse the data with any SQLite client, or from the command line:

```
sqlite3 football.db "SELECT * FROM ratings WHERE season='2627' AND round=11 ORDER BY att_mean DESC"
```

## Known limitations

- **The backtests haven't been moved to the database.** The scripts in `backtest/` still read the old CSV and JSON files and call `w_bhm.py` with an outdated command line, so they won't run as they are.
- **2023–26 has no round numbers.** Those seasons are stored with match dates only, because the source CSVs don't include gameweeks.
- **Postponed matches.** A postponed match stays in its original round. `w_bhm.py` warns about it, but its result is skipped once that round has been processed.
- **No reset between seasons.** Ratings carried over from last season are not shrunk toward 0 and their variances are not widened before round 1, so early-season ratings may be over-confident.
- **The intercept is fixed** and doesn't learn from data, so a league-wide change in scoring rate isn't picked up.
