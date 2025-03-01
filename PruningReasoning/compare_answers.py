#!/usr/bin/env python3
import json
import csv
import os
import glob
import argparse
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
                
                # Store all extracted answers for this problem
                extracted_answers[problem_id] = [
                    result.get('extracted_answer') for result in results
                ]
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return extracted_answers

def compare_answers(ground_truth, extracted_answers):
    """Compare extracted answers with ground truth and calculate accuracy."""
    results = {
        'total_problems': 0,
        'problems_with_matches': 0,
        'total_responses': 0,
        'correct_responses': 0,
        'problem_details': {}
    }
    
    for problem_id, truth_answer in ground_truth.items():
        if problem_id in extracted_answers:
            results['total_problems'] += 1
            extracted = extracted_answers[problem_id]
            
            # Count correct responses for this problem
            correct_count = sum(1 for ans in extracted if ans == truth_answer)
            total_count = len(extracted)
            
            results['total_responses'] += total_count
            results['correct_responses'] += correct_count
            
            # Record if at least one response was correct
            has_match = correct_count > 0
            if has_match:
                results['problems_with_matches'] += 1
            
            # Store details for this problem
            results['problem_details'][problem_id] = {
                'ground_truth': truth_answer,
                'extracted_answers': extracted,
                'correct_count': correct_count,
                'total_count': total_count,
                'accuracy': correct_count / total_count if total_count > 0 else 0
            }
    
    # Calculate overall accuracy
    results['response_accuracy'] = (
        results['correct_responses'] / results['total_responses'] 
        if results['total_responses'] > 0 else 0
    )
    
    results['problem_accuracy'] = (
        results['problems_with_matches'] / results['total_problems']
        if results['total_problems'] > 0 else 0
    )
    
    return results

def generate_report(results, output_file=None):
    """Generate a detailed report of the comparison results."""
    report = []
    
    # Overall statistics
    report.append("# Answer Comparison Report")
    report.append("")
    report.append("## Overall Statistics")
    report.append("")
    report.append(f"- Total problems analyzed: {results['total_problems']}")
    report.append(f"- Problems with at least one correct answer: {results['problems_with_matches']} ({results['problem_accuracy']:.2%})")
    report.append(f"- Total responses analyzed: {results['total_responses']}")
    report.append(f"- Correct responses: {results['correct_responses']} ({results['response_accuracy']:.2%})")
    report.append("")
    
    # Problem-specific details
    report.append("## Problem Details")
    report.append("")
    
    # Sort problems by ID for consistent reporting
    sorted_problems = sorted(results['problem_details'].items())
    
    for problem_id, details in sorted_problems:
        report.append(f"### Problem {problem_id}")
        report.append("")
        report.append(f"- Ground truth answer: {details['ground_truth']}")
        report.append(f"- Extracted answers: {details['extracted_answers']}")
        report.append(f"- Accuracy: {details['correct_count']}/{details['total_count']} ({details['accuracy']:.2%})")
        report.append("")
    
    report_text = "\n".join(report)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description='Compare extracted answers with ground truth')
    parser.add_argument('--csv', type=str, default='AIME_2023_4.csv', help='Path to the CSV file with ground truth answers')
    parser.add_argument('--extracted', type=str, default='extracted_answers', help='Directory containing extracted answer files')
    parser.add_argument('--output', type=str, default='comparison_report.md', help='Output file for the comparison report')
    args = parser.parse_args()
    
    # Load ground truth and extracted answers
    ground_truth = load_ground_truth(args.csv)
    extracted_answers = load_extracted_answers(args.extracted)
    
    # Compare answers
    results = compare_answers(ground_truth, extracted_answers)
    
    # Generate and save report
    report = generate_report(results, args.output)
    
    # Print summary to console
    print(f"Comparison complete. Overall accuracy: {results['response_accuracy']:.2%}")
    print(f"Problems with at least one correct answer: {results['problems_with_matches']}/{results['total_problems']} ({results['problem_accuracy']:.2%})")
    print(f"Detailed report saved to {args.output}")

if __name__ == "__main__":
    main() 