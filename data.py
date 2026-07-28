import pandas as pd
import json
import os

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

if __name__ == "__main__":
    # Reference the raw file verbatim
    prepare_match_data("data/season-2526.csv")