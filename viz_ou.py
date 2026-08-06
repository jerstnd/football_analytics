import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize_ou_backtest(csv_path):
    if not os.path.exists(csv_path):
        print(f"[Error] Could not find the results file: {csv_path}")
        sys.exit(1)
        
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['Gameweek', 'True O/U', 'Pred O/U', 'Log Loss', 'Brier Score']
    for col in required_cols:
        if col not in df.columns:
            print(f"[Error] Missing expected column '{col}' in CSV.")
            sys.exit(1)

    # 1. Data Prep
    df['Correct'] = (df['True O/U'] == df['Pred O/U']).astype(int)
    df['Match_Number'] = range(1, len(df) + 1)
    df['Cumulative_Accuracy'] = (df['Correct'].cumsum() / df['Match_Number']) * 100
    
    # Aggregate error metrics by Gameweek
    gw_stats = df.groupby('Gameweek').agg(
        LogLoss=('Log Loss', 'mean'),
        BrierScore=('Brier Score', 'mean')
    ).reset_index()

    # 2. Setup the Dashboard Canvas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('O/U 2.5 Backtest Diagnostic Dashboard', fontsize=16, fontweight='bold')

    # --- PANEL 1: Cumulative Accuracy ---
    axes[0, 0].plot(df['Match_Number'], df['Cumulative_Accuracy'], color='blue', linewidth=2)
    axes[0, 0].axhline(50, color='red', linestyle='--', alpha=0.7, label='Random Coin Flip (50%)')
    axes[0, 0].set_title('Cumulative Accuracy Over the Season')
    axes[0, 0].set_xlabel('Match Number')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(30, 70)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- PANEL 2: Error Metrics (Log Loss & Brier Score) ---
    axes[0, 1].plot(gw_stats['Gameweek'], gw_stats['LogLoss'], color='darkorange', marker='o', label='Mean Log Loss')
    axes[0, 1].plot(gw_stats['Gameweek'], gw_stats['BrierScore'], color='purple', marker='s', label='Mean Brier Score')
    axes[0, 1].set_title('Error Metrics by Gameweek (Lower is Better)')
    axes[0, 1].set_xlabel('Gameweek')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- PANEL 3: Prediction Imbalance (The "Bug Finder") ---
    pred_counts = df['Pred O/U'].value_counts()
    true_counts = df['True O/U'].value_counts()
    
    # Ensure both keys exist even if model predicted 0 of one outcome
    for outcome in ['Over', 'Under']:
        if outcome not in pred_counts: pred_counts[outcome] = 0
        if outcome not in true_counts: true_counts[outcome] = 0

    x = [0, 1]
    width = 0.35
    axes[1, 0].bar([p - width/2 for p in x], [pred_counts['Over'], pred_counts['Under']], width, label='Model Predicted', color='dodgerblue')
    axes[1, 0].bar([p + width/2 for p in x], [true_counts['Over'], true_counts['Under']], width, label='Actual Outcomes', color='slategray')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Over', 'Under'])
    axes[1, 0].set_title('Model Predictions vs. Reality (Total Counts)')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # --- PANEL 4: Rolling 20-Match Accuracy (Momentum) ---
    df['Rolling_Acc'] = df['Correct'].rolling(window=20).mean() * 100
    axes[1, 1].plot(df['Match_Number'], df['Rolling_Acc'], color='forestgreen', linewidth=1.5)
    axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Rolling 20-Match Accuracy')
    axes[1, 1].set_xlabel('Match Number')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_ylim(20, 80)
    axes[1, 1].grid(True, alpha=0.3)

    # 3. Finalize and Output
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leave room for the main title
    
    output_filename = csv_path.replace('.csv', '_dashboard.png')
    plt.savefig(output_filename, dpi=300)
    print(f"\nSuccess! Dashboard saved as a high-res image to: {output_filename}")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diagnostic visualizations for the O/U Backtest.")
    parser.add_argument("--csv", default="prediction/backtest_ou2_5_results.csv", help="Path to your backtest results CSV")
    
    args = parser.parse_args()
    visualize_ou_backtest(args.csv)