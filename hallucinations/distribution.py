import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set styling
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Read the CSV file
file_path = 'hallucination_experiment_results/gemini_uncap_results.csv'
df = pd.read_csv(file_path)

# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Diversity Metrics in Gemini Responses', fontsize=16)

# Histogram of reasoning diversity
sns.histplot(df['reasoning_diversity'], kde=True, ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Reasoning Diversity Distribution')
axs[0, 0].set_xlabel('Reasoning Diversity Score')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].axvline(df['reasoning_diversity'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["reasoning_diversity"].mean():.2f}')
axs[0, 0].legend()

# Histogram of final diversity
sns.histplot(df['final_diversity'], kde=True, ax=axs[0, 1], color='green')
axs[0, 1].set_title('Final Diversity Distribution')
axs[0, 1].set_xlabel('Final Diversity Score')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].axvline(df['final_diversity'].mean(), color='red', linestyle='--', 
                 label=f'Mean: {df["final_diversity"].mean():.2f}')
axs[0, 1].legend()

# Scatter plot: reasoning vs final diversity
sns.scatterplot(x='reasoning_diversity', y='final_diversity', data=df, ax=axs[1, 0], alpha=0.7)
axs[1, 0].set_title('Reasoning vs Final Diversity')
axs[1, 0].set_xlabel('Reasoning Diversity Score')
axs[1, 0].set_ylabel('Final Diversity Score')

# Add regression line
sns.regplot(x='reasoning_diversity', y='final_diversity', data=df, 
            scatter=False, ax=axs[1, 0], color='red', line_kws={"linestyle": "--"})

# Correlation coefficient
corr = df['reasoning_diversity'].corr(df['final_diversity'])
axs[1, 0].annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')

# Box plot comparison
plt_data = pd.melt(df[['reasoning_diversity', 'final_diversity']], 
                  var_name='Metric', value_name='Score')
sns.boxplot(x='Metric', y='Score', data=plt_data, ax=axs[1, 1])
axs[1, 1].set_title('Comparison of Diversity Metrics')
axs[1, 1].set_xlabel('')
axs[1, 1].set_ylabel('Score')

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

axs[1, 1].annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                  bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
plt.savefig('gemini_diversity_analysis.png', dpi=300, bbox_inches='tight')

# Display the plots
plt.show()

print("Analysis complete! Visualization saved as 'gemini_diversity_analysis.png'")