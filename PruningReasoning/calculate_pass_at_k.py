#!/usr/bin/env python3
import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import csv
from collections import defaultdict

def load_ground_truth(csv_file):
    """Load ground truth answers from the CSV file."""
    ground_truth = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create a key in the format "year_problem_number"
            key = f"{row['Year']}_{row['Problem Number']}"
            ground_truth[key] = row['Answer']
    return ground_truth

def load_extracted_answers(directory):
    """Load all extracted answer files from the specified directory."""
    extracted_answers = {}
    files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                problem_id = data.get('problem_id')
                results = data.get('results', [])
                
                # Store all extracted answers for this problem with their response index
                extracted_answers[problem_id] = {
                    result.get('response_index'): result.get('extracted_answer') 
                    for result in results
                }
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return extracted_answers

def calculate_correct_counts(extracted_answers, ground_truth):
    """Calculate the number of correct answers for each problem."""
    correct_counts = {}
    
    for problem_id, answers in extracted_answers.items():
        if problem_id not in ground_truth:
            print(f"Warning: No ground truth available for problem {problem_id}")
            continue
            
        truth = ground_truth[problem_id]
        correct = sum(1 for answer in answers.values() if answer == truth)
        correct_counts[problem_id] = correct
    
    return correct_counts

def calculate_pass_at_k(correct_counts, n, k_values):
    """
    Calculate pass@k using the unbiased estimation formula:
    
    pass@k = (1 / # of problems) * sum_{i=1}^{# of problems} (1 - ((N-C_i) choose k) / (N choose k))
    
    Where:
    - N is the total number of samples per problem
    - C_i is the number of correct samples for problem i
    - k is the number of samples to consider
    """
    num_problems = len(correct_counts)
    if num_problems == 0:
        return {k: 0.0 for k in k_values}
    
    pass_at_k = {}
    
    for k in k_values:
        if k > n:
            print(f"Warning: k ({k}) cannot be greater than n ({n})")
            pass_at_k[k] = 0.0
            continue
            
        # Calculate the denominator (N choose k) once
        n_choose_k = comb(n, k, exact=True)
        
        # Calculate the sum in the formula
        sum_term = 0.0
        for c_i in correct_counts.values():
            if(k == 1):
                print(c_i)
            # Calculate ((N-C_i) choose k) / (N choose k)
            n_minus_c_i_choose_k = comb(n - c_i, k, exact=True)
            probability = 1.0 - (n_minus_c_i_choose_k / n_choose_k)
            sum_term += probability
            # If c_i = 0, the probability is 0.0 (no contribution to sum)
        
        # Calculate the final pass@k value
        pass_at_k[k] = sum_term / num_problems
    
    return pass_at_k

def plot_pass_at_k(pass_at_k_values, output_file=None, max_k=50, is_original=True):
    """
    Plot the pass@k values up to max_k.
    Shows the monotonically increasing nature of the pass@k metric.
    
    Args:
        pass_at_k_values: Dictionary mapping k values to pass@k values
        output_file: Output file path for the plot
        max_k: Maximum k value to plot
        is_original: Whether this is the original dataset (affects title)
    """
    # Filter k values up to max_k
    k_values = sorted([k for k in pass_at_k_values.keys() if k <= max_k])
    pass_values = [pass_at_k_values[k] for k in k_values]
    
    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    
    # Plot the line
    plt.plot(k_values, pass_values, marker='o', linestyle='-', linewidth=2, color='#1f77b4')
    
    # Add value labels for specific points with improved positioning
    for i, (k, v) in enumerate(zip(k_values, pass_values)):
        # Only label some points to avoid clutter
        if k == 1 or k == 5 or k == 10 or k == 25 or k == 50 or i == len(k_values) - 1:
            # Alternate between top and bottom positioning to avoid overlap
            if k % 10 == 0:
                y_offset = -20  # Place below the point
                va = 'top'
            else:
                y_offset = 15  # Place above the point
                va = 'bottom'
                
            plt.annotate(f'{v:.3f}', (k, v), textcoords="offset points", 
                        xytext=(0, y_offset), ha='center', va=va, fontsize=10)
    
    # Set axis labels and title with specific dataset information
    if is_original:
        title = 'Baseline pass@k Performance on AIME 2023 I and 2024 I'
    else:
        title = 'Baseline pass@k Performance on AIME 2023 II'
    
    plt.xlabel('k (Number of Samples)', fontsize=14)
    plt.ylabel('pass@k', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Set y-axis limits with some padding
    # Start y-axis 0.2 below the lowest value, but not below 0
    min_value = min(pass_values)
    y_min = max(0, min_value - 0.2)
    plt.ylim(y_min, min(1.05, max(pass_values) * 1.1))
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()  # Standard tight layout without special adjustments
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate pass@k metrics for extracted answers')
    parser.add_argument('--csv', type=str, default='AIME_2023_4.csv', help='Path to the CSV file with ground truth answers')
    parser.add_argument('--extracted', type=str, default='extracted_answers', help='Directory containing extracted answer files')
    parser.add_argument('--n', type=int, default=50, help='Total number of samples per problem')
    parser.add_argument('--max_k', type=int, default=50, help='Maximum k value to calculate and plot')
    parser.add_argument('--output', type=str, default='pass_at_k.png', help='Output file for the pass@k plot')
    parser.add_argument('--is_original', action='store_true', help='Whether this is the original dataset (affects title)')
    args = parser.parse_args()
    
    # Generate k values from 1 to max_k
    k_values = list(range(1, args.max_k + 1))
    print(f"Calculating pass@k for k values from 1 to {args.max_k}")
    
    # Load ground truth and extracted answers
    print("Loading ground truth answers...")
    ground_truth = load_ground_truth(args.csv)
    
    print("Loading extracted answers...")
    extracted_answers = load_extracted_answers(args.extracted)
    
    # Calculate the number of correct answers for each problem
    print("Calculating correct counts...")
    correct_counts = calculate_correct_counts(extracted_answers, ground_truth)
    
    # Print summary of correct counts
    print("\nCorrect answers summary:")
    for problem_id, count in correct_counts.items():
        total = len(extracted_answers.get(problem_id, {}))
        print(f"Problem {problem_id}: {count}/{total} correct answers ({count/total:.1%})")
    
    # Calculate pass@k values
    print("\nCalculating pass@k values...")
    pass_at_k_values = calculate_pass_at_k(correct_counts, args.n, k_values)
    
    # Print the results for key values
    print("\npass@k Results:")
    for k in [1, 5, 10, 25, 50]:
        if k in pass_at_k_values:
            print(f"pass@{k}: {pass_at_k_values[k]:.4f}")
    
    # Plot the results
    print("\nGenerating plot...")
    plot_pass_at_k(pass_at_k_values, args.output, args.max_k, args.is_original)
    
    print(f"Analysis complete. Plot saved to {args.output}")

if __name__ == "__main__":
    main() 