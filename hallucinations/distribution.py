import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate diversity metric visualizations')
    parser.add_argument('--plot', type=str, required=True, 
                        choices=['reasoning_hist', 'final_hist', 'scatter', 'boxplot'],
                        help='Which plot to generate')
    parser.add_argument('--file', type=str, default='hallucination_experiment_results/gemini_250_results.csv',
                        help='Path to CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Set styling
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    # Read the CSV file
    df = pd.read_csv(args.file)
    
    # Create figure based on selected plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if args.plot == 'reasoning_hist':
        # Histogram of reasoning diversity
        sns.histplot(df['reasoning_diversity'], kde=True, ax=ax, color='blue')
        ax.set_title('Reasoning Diversity Distribution')
        ax.set_xlabel('Reasoning Diversity Score')
        ax.set_ylabel('Frequency')
        ax.axvline(df['reasoning_diversity'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["reasoning_diversity"].mean():.2f}')
        ax.legend()
        
    elif args.plot == 'final_hist':
        # Histogram of final diversity
        sns.histplot(df['final_diversity'], kde=True, ax=ax, color='green')
        ax.set_title('Final Diversity Distribution')
        ax.set_xlabel('Final Diversity Score')
        ax.set_ylabel('Frequency')
        ax.axvline(df['final_diversity'].mean(), color='red', linestyle='--', 
                 label=f'Mean: {df["final_diversity"].mean():.2f}')
        ax.legend()
        
    elif args.plot == 'scatter':
        # Scatter plot: reasoning vs final diversity
        sns.scatterplot(x='reasoning_diversity', y='final_diversity', data=df, ax=ax, alpha=0.7)
        ax.set_title('Reasoning vs Final Diversity')
        ax.set_xlabel('Reasoning Diversity Score')
        ax.set_ylabel('Final Diversity Score')
        
        # Add regression line
        sns.regplot(x='reasoning_diversity', y='final_diversity', data=df, 
                    scatter=False, ax=ax, color='red', line_kws={"linestyle": "--"})
        
        # Correlation coefficient
        corr = df['reasoning_diversity'].corr(df['final_diversity'])
        ax.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
    elif args.plot == 'boxplot':
        # Box plot comparison
        plt_data = pd.melt(df[['reasoning_diversity', 'final_diversity']], 
                          var_name='Metric', value_name='Score')
        sns.boxplot(x='Metric', y='Score', data=plt_data, ax=ax)
        ax.set_title('Comparison of Diversity Metrics')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        
        # Add summary statistics as text
        summary_stats = df[['reasoning_diversity', 'final_diversity']].describe().round(2)
        stats_text = (f"Reasoning Diversity:\n"
                      f"  Mean: {summary_stats.loc['mean', 'reasoning_diversity']}\n"
                      f"  Median: {summary_stats.loc['50%', 'reasoning_diversity']}\n"
                      f"  Std Dev: {summary_stats.loc['std', 'reasoning_diversity']}\n\n"
                      f"Final Diversity:\n"
                      f"  Mean: {summary_stats.loc['mean', 'final_diversity']}\n"
                      f"  Median: {summary_stats.loc['50%', 'final_diversity']}\n"
                      f"  Std Dev: {summary_stats.loc['std', 'final_diversity']}")
        
        ax.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                      bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Determine output filename if not provided
    if args.output is None:
        output_file = f"{args.plot}_{args.file.split('/')[-1].replace('.csv', '')}.png"
    else:
        output_file = args.output
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    print(f"Plot saved as '{output_file}'")

if __name__ == "__main__":
    main()