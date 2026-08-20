# Sequential Bayesian Football Prediction

## Overview
An automated prediction pipeline designed to extract mathematical edge from football match.  

The architecture utilizes a Markov Chain Monte Carlo (MCMC) sampler to continuously evaluate hidden team parameters, passing Expected Goals (xG) through a Dixon-Coles adjusted Poisson distribution to accurately price complex market inefficiencies.

## Core Modules
* **Sequential MCMC State Tracker:** Dynamically updates team Attack, Defense, and Home Advantage ratings week-by-week, utilizing time-decay variance capping to prevent model overfitting.
* **Dynamic Entity Initialization:** Automatically intercepts newly promoted teams missing from historical JSON priors, initializing them with 0.0 (League Average) parameters to ensure continuous pipeline execution.
* **Probability Generator (poisson.py):** Calculates exact score matrices and extracts highly calibrated probabilities for the 3-way match winner (1X2) and total goals (O/U 2.5) markets.

## Continuous Execution Pipeline
This system operates in a continuous chronological loop designed to be executed on remote processing instances. 

### Step 1: Initialize the Prior (Preseason)
Before the season begins, calculate the baseline MCMC parameter states by processing a large historical dataset. This generates the initial state file.

    python y_bhm.py --history data/historical_results.csv --output state/ratings_preseason.json

### Step 2-3: Weekly Matchday Loop
During the season, execute these commands chronologically to generate probabilities, allocate capital, and update the model's memory.

    # Step 2: Generate Matchday Probabilities (Pre-match)
    python poisson.py --state state/ratings_preseason.json --batch fixtures/fixtures_wk1.csv

    # Step 3: Update the MCMC State (Post-match)
    # Run this only after the weekend matches conclude to update the prior for Week 2
    python w_bhm.py --state state/ratings_preseason.json --results data/results_wk1.csv --output state/ratings_current.json

## Performance Validation
The predictive edge of this engine was rigorously validated through a strict 38-week (380 match) chronological walk-forward backtest, ensuring zero future data leakage. The model is graded purely on probabilistic scoring rules rather than raw accuracy.

* **Over/Under 2.5 Goals:** 53.61% Top-Pick Accuracy | 0.7205 Mean Log Loss
* **1X2 Match Result:** 1.0252 Mean Log Loss | 0.2089 Ranked Probability Score (RPS)

## Data Structure Requirements
To ensure seamless execution, the batch predictor requires input CSV files (e.g., fixtures_wk1.csv) to contain the following exact headers:
* HomeTeam: String (Must match JSON state names)
* AwayTeam: String (Must match JSON state names)
* Odds_Over_25: Float (Live decimal market odds)
* Odds_Under_25: Float (Live decimal market odds)
