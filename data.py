import os
import sys
import json
import argparse
import pandas as pd

def prepare_match_data(input_csv, output_dir="data"):
    print(f"Reading raw data from: {input_csv}")
    
    # 1. Load the raw CSV
    df = pd.read_csv(input_csv)
    
    # 2. Extract ONLY the 4 required columns
    # FTHG = Full Time Home Goals, FTAG = Full Time Away Goals
    df_filtered = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
    
    # 3. Create a unique ID for every team in the league
    unique_teams = pd.concat([df_filtered['HomeTeam'], df_filtered['AwayTeam']]).unique()
    unique_teams.sort()
    
    team_to_id = {team: idx for idx, team in enumerate(unique_teams)}
    id_to_team = {idx: team for idx, team in enumerate(unique_teams)}
    
    # 4. Map the strings to integer IDs
    df_filtered['home_id'] = df_filtered['HomeTeam'].map(team_to_id)
    df_filtered['away_id'] = df_filtered['AwayTeam'].map(team_to_id)
    
    # Rename goals columns to match our MCMC script expectations
    df_filtered = df_filtered.rename(columns={'FTHG': 'home_goals', 'FTAG': 'away_goals'})
    
    # Finalize the 4 required columns for the MCMC model
    df_final = df_filtered[['home_id', 'away_id', 'home_goals', 'away_goals']]
    
    # 5. Save the output
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the cleaned CSV for the model
    output_csv = os.path.join(output_dir, 'matches_ready.csv')
    df_final.to_csv(output_csv, index=False)
    
    # Save the Team ID mapping so we know which ID belongs to which team
    mapping_file = os.path.join(output_dir, 'team_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(id_to_team, f, indent=4)
        
    print(f"Success! Cleaned matches saved to: {output_csv}")
    print(f"Team ID mapping saved to: {mapping_file}")

def clean_historical_seasons(csv_files, mapping_path="data/team_mapping.json", output_csv="data/train_23_25_ready.csv"):
    os.makedirs("data", exist_ok=True)

    # 1. Load existing team mapping if it exists
    if os.path.exists(mapping_path):
        print(f"Loading existing mapping from: {mapping_path}")
        with open(mapping_path, "r") as f:
            # Keys in JSON are strings; convert to int -> name, then invert to name -> int
            id_to_name = {int(k): v for k, v in json.load(f).items()}
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        next_id = max(id_to_name.keys()) + 1 if id_to_name else 0
    else:
        print(f"[Notice] No existing {mapping_path} found. Starting fresh mapping from ID 0.")
        name_to_id = {}
        id_to_name = {}
        next_id = 0

    cleaned_dfs = []

    # 2. Process each historical CSV file
    for filepath in csv_files:
        if not os.path.exists(filepath):
            print(f"[Error] File not found: {filepath}")
            sys.exit(1)

        print(f"Processing historical season: {filepath}")
        df = pd.read_csv(filepath)

        # Standardize column names (handles both 'FTHG'/'FTAG' and 'home_goals'/'away_goals')
        col_map = {}
        if 'FTHG' in df.columns:
            col_map = {'FTHG': 'home_goals', 'FTAG': 'away_goals'}
        df = df.rename(columns=col_map)

        df_filtered = df[['HomeTeam', 'AwayTeam', 'home_goals', 'away_goals']].copy()

        # 3. Check for any unmapped teams (relegated teams) and assign IDs sequentially
        season_teams = set(df_filtered['HomeTeam']).union(set(df_filtered['AwayTeam']))
        for team in sorted(season_teams):
            if team not in name_to_id:
                print(f"  [+] New historical team discovered: '{team}' -> Assigned ID {next_id}")
                name_to_id[team] = next_id
                id_to_name[next_id] = team
                next_id += 1

        # 4. Map names to consistent IDs
        df_filtered['home_id'] = df_filtered['HomeTeam'].map(name_to_id)
        df_filtered['away_id'] = df_filtered['AwayTeam'].map(name_to_id)

        # Reorder to standard schema
        df_final = df_filtered[['home_id', 'away_id', 'home_goals', 'away_goals']]
        cleaned_dfs.append(df_final)

    # 5. Concatenate all seasons into one master training CSV
    master_df = pd.concat(cleaned_dfs, ignore_index=True)
    master_df.to_csv(output_csv, index=False)

    # 6. Save the expanded mapping back to JSON
    with open(mapping_path, "w") as f:
        json.dump(id_to_name, f, indent=4)

    print("\n" + "="*70)
    print(" HISTORICAL DATA CLEANING COMPLETE")
    print("="*70)
    print(f"Total Matches Combined : {len(master_df)}")
    print(f"Total Unique Teams     : {len(id_to_name)} (including relegated teams)")
    print(f"Saved Cleaned Data     : {output_csv}")
    print(f"Updated Mapping File   : {mapping_path}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Engineering Pipeline for Bayesian Football Model")
    
    parser.add_argument("--history", nargs="+", help="Run historical cleaner on a list of CSV files (e.g., season-2324.csv season-2425.csv)")
    parser.add_argument("--prep", help="Run standard single-season preparation on a CSV file (e.g., data/season-2526.csv)")
    parser.add_argument("--out", default="data/train_23_25_ready.csv", help="Output CSV path when using --history")
    parser.add_argument("--mapping", default="data/team_mapping.json", help="Path to team_mapping.json")

    args = parser.parse_args()

    if args.history:
        # Runs the historical cleaner and preserves/extends existing IDs
        clean_historical_seasons(args.history, args.mapping, args.out)
    elif args.prep:
        # Runs standard preparation on a single season CSV
        prepare_match_data(args.prep)
    else:
        # Fallback to your original default behavior if no flags are provided
        print("[Notice] No arguments provided. Running default preparation on 'data/season-2526.csv'...")
        prepare_match_data("data/season-2526.csv")