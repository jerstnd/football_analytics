import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def generate_dashboard(csv_path, output_path):
    print(f"Reading backtest data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Ensure required columns exist based on your specific backtest.py output
    required_cols = ['True Result', 'Pred Result', 'Log Loss', 'RPS']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. Check your backtest CSV headers.")

    # 2. Process Core Metrics
    # Calculate if the model's top pick was correct
    df['Correct'] = (df['Pred Result'] == df['True Result']).astype(int)

    # 3. Calculate Cumulative Metrics for smooth line plotting
    df['Cumulative_Accuracy'] = df['Correct'].expanding().mean()
    df['Cumulative_Log_Loss'] = df['Log Loss'].expanding().mean()
    df['Cumulative_RPS'] = df['RPS'].expanding().mean()

    # 4. Setup the Matplotlib Dashboard (2x2 Grid)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('1X2 Bayesian Trading Engine: Backtest Dashboard', fontsize=18, fontweight='bold', y=0.95)
    
    # --- Panel 1: Cumulative Accuracy ---
    axs[0, 0].plot(df.index, df['Cumulative_Accuracy'], color='#2196F3', linewidth=2)
    axs[0, 0].axhline(y=0.45, color='gray', linestyle='--', alpha=0.7, label='Naive Home Guess (~45%)')
    axs[0, 0].set_title('Cumulative Top-Pick Accuracy')
    axs[0, 0].set_xlabel('Matches Played')
    axs[0, 0].set_ylabel('Accuracy Rate')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # --- Panel 2: Cumulative RPS ---
    axs[0, 1].plot(df.index, df['Cumulative_RPS'], color='#9C27B0', linewidth=2)
    axs[0, 1].set_title('Cumulative Ranked Probability Score (RPS)')
    axs[0, 1].set_xlabel('Matches Played')
    axs[0, 1].set_ylabel('RPS (Lower is Better)')
    axs[0, 1].grid(True, alpha=0.3)

    # --- Panel 3: Cumulative Log Loss ---
    axs[1, 0].plot(df.index, df['Cumulative_Log_Loss'], color='#F44336', linewidth=2)
    axs[1, 0].axhline(y=1.066, color='gray', linestyle='--', alpha=0.7, label='League Base Rate (1.066)')
    axs[1, 0].axhline(y=1.098, color='black', linestyle=':', alpha=0.7, label='Blind Guess (1.098)')
    axs[1, 0].set_title('Cumulative Log Loss')
    axs[1, 0].set_xlabel('Matches Played')
    axs[1, 0].set_ylabel('Log Loss (Lower is Better)')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # --- Panel 4: Model Predictions vs Reality (Bar Chart) ---
    pred_counts = df['Pred Result'].value_counts().reindex(['H', 'D', 'A'], fill_value=0)
    actual_counts = df['True Result'].value_counts().reindex(['H', 'D', 'A'], fill_value=0)
    
    x = np.arange(3)
    width = 0.35
    
    axs[1, 1].bar(x - width/2, pred_counts, width, label='Model Predicted', color='#1E90FF')
    axs[1, 1].bar(x + width/2, actual_counts, width, label='Actual Outcomes', color='#708090')
    
    axs[1, 1].set_title('Model Predictions vs. Reality (Total Counts)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(['Home', 'Draw', 'Away'])
    axs[1, 1].set_ylabel('Number of Matches')
    axs[1, 1].grid(axis='y', alpha=0.3)
    axs[1, 1].legend()

    # 5. Finalize and Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard successfully generated and saved to: {output_path}")
    
    # Print Final Terminal Summary
    print("\n" + "="*40)
    print("SEASON CUMULATIVE PERFORMANCE")
    print("="*40)
    print(f"Total Matches Evaluated : {len(df)}")
    print(f"Overall Top-Pick Acc    : {df['Cumulative_Accuracy'].iloc[-1]:.2%}")
    print("-" * 40)
    print(f"Overall Mean Log Loss   : {df['Cumulative_Log_Loss'].iloc[-1]:.4f}")
    print(f"Overall Mean RPS        : {df['Cumulative_RPS'].iloc[-1]:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1X2 Backtest Dashboard")
    parser.add_argument("--csv", type=str, required=True, help="Path to the backtest CSV results")
    parser.add_argument("--output", type=str, default="prediction/dashboard_1x2.png", help="Path to save the generated PNG dashboard")
    
    args = parser.parse_args()
    generate_dashboard(args.csv, args.output)