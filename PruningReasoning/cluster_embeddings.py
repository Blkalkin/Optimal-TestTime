#!/usr/bin/env python3
import json
import os
import glob
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def load_embeddings(directory):
    """Load all embedding files from the specified directory."""
    embeddings_data = {}
    files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                problem_id = data.get('problem_id')
                embeddings_data[problem_id] = data
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return embeddings_data

def organize_embeddings_by_chunk(problem_data):
    """Organize embeddings by chunk index across all responses."""
    chunk_embeddings = defaultdict(list)
    response_indices = defaultdict(list)
    
    for response in problem_data.get('responses', []):
        response_idx = response.get('response_idx')
        embeddings = response.get('embeddings', [])
        
        # Check if chunk_indices field exists (from updated chunk_and_embed.py)
        chunk_indices = response.get('chunk_indices')
        
        if chunk_indices:
            # Use the provided chunk_indices
            for i, (chunk_idx, embedding) in enumerate(zip(chunk_indices, embeddings)):
                chunk_embeddings[chunk_idx].append(embedding)
                response_indices[chunk_idx].append(response_idx)
        else:
            # Fallback to the old method (assuming sequential chunk indices)
            for chunk_idx, embedding in enumerate(embeddings):
                chunk_embeddings[chunk_idx].append(embedding)
                response_indices[chunk_idx].append(response_idx)
    
    return chunk_embeddings, response_indices

def limit_to_midway(chunk_embeddings, response_indices, specific_chunks=None):
    """
    Limit chunks to process only up to the midway point or to specific chunks if provided.
    
    Args:
        chunk_embeddings: Dictionary mapping chunk indices to embeddings
        response_indices: Dictionary mapping chunk indices to response indices
        specific_chunks: List of specific chunk indices to keep (e.g., [0, 1, 2, 3])
    """
    if not chunk_embeddings:
        return {}, {}
    
    if specific_chunks is not None:
        # Filter to only include the specified chunks
        limited_chunk_embeddings = {
            idx: embeddings for idx, embeddings in chunk_embeddings.items() 
            if idx in specific_chunks
        }
        
        limited_response_indices = {
            idx: indices for idx, indices in response_indices.items() 
            if idx in specific_chunks
        }
        
        print(f"Filtered to specific chunks: {sorted(limited_chunk_embeddings.keys())}")
        return limited_chunk_embeddings, limited_response_indices
    else:
        # Find the maximum chunk index
        max_chunk_idx = max(chunk_embeddings.keys())
        
        # Calculate the midway point
        midway = max_chunk_idx // 2
        
        # Filter chunks up to the midway point
        limited_chunk_embeddings = {
            idx: embeddings for idx, embeddings in chunk_embeddings.items() 
            if idx <= midway
        }
        
        limited_response_indices = {
            idx: indices for idx, indices in response_indices.items() 
            if idx <= midway
        }
        
        return limited_chunk_embeddings, limited_response_indices

def calculate_silhouette_scores(embeddings, cluster_sizes):
    """Calculate silhouette scores for different numbers of clusters."""
    silhouette_scores = {}
    embeddings_array = np.array(embeddings)
    
    # Need at least 2 samples and 2 clusters for silhouette score
    if len(embeddings) < 2:
        return silhouette_scores
    
    for n_clusters in cluster_sizes:
        # Skip if too many clusters requested or only one cluster
        if n_clusters >= len(embeddings) or n_clusters < 2:
            continue
            
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(embeddings_array, cluster_labels)
            silhouette_scores[n_clusters] = score
        except Exception as e:
            print(f"Error calculating silhouette score for {n_clusters} clusters: {str(e)}")
    
    return silhouette_scores

def cluster_embeddings(embeddings, n_clusters):
    """Perform k-means clustering on embeddings."""
    if len(embeddings) < n_clusters:
        # If we have fewer embeddings than requested clusters, adjust n_clusters
        n_clusters = max(1, len(embeddings))
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings_array)
    
    # Get cluster assignments and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

def find_closest_to_centroid(embeddings, centroid):
    """Find the index of the embedding closest to the centroid."""
    distances = np.linalg.norm(np.array(embeddings) - centroid, axis=1)
    return np.argmin(distances)

def evaluate_clustering(problem_id, chunk_embeddings, response_indices, extracted_answers, ground_truth, cluster_sizes):
    """Evaluate clustering for different numbers of clusters."""
    results = {
        'problem_id': problem_id,
        'ground_truth': ground_truth.get(problem_id),
        'chunk_results': {},
        'summary': {},
        'silhouette_scores': {}
    }
    
    # Skip if no ground truth available
    if problem_id not in ground_truth:
        print(f"No ground truth available for problem {problem_id}, skipping...")
        return None
    
    truth_answer = ground_truth[problem_id]
    
    # Process each chunk index
    print(f"Total chunks: {len(chunk_embeddings)}")
    for chunk_idx, embeddings in chunk_embeddings.items():
        chunk_results = {}
        
        # Skip if too few embeddings
        if len(embeddings) < 2:
            continue
        
        # Calculate silhouette scores for this chunk
        silhouette_scores = calculate_silhouette_scores(embeddings, cluster_sizes)
        
        # Get response indices for this chunk
        resp_indices = response_indices[chunk_idx]
        
        # Evaluate for different cluster sizes
        for n_clusters in cluster_sizes:
            print(f"Evaluating {n_clusters} clusters for problem {problem_id} chunk {chunk_idx}")
            # Skip if too many clusters requested
            if n_clusters > len(embeddings):
                continue
                
            # Cluster the embeddings
            labels, centroids = cluster_embeddings(embeddings, n_clusters)
            
            # Find responses closest to centroids
            selected_indices = []
            for centroid in centroids:
                closest_idx = find_closest_to_centroid(embeddings, centroid)
                selected_indices.append(resp_indices[closest_idx])
            
            # Check if any selected response has the correct answer
            has_correct = False
            for idx in selected_indices:
                if idx in extracted_answers.get(problem_id, {}) and extracted_answers[problem_id][idx] == truth_answer:
                    has_correct = True
                    break
            
            chunk_results[n_clusters] = {
                'selected_indices': selected_indices,
                'has_correct_answer': has_correct
            }
        
        results['chunk_results'][chunk_idx] = chunk_results
        
        # Store silhouette scores for this chunk
        if silhouette_scores:
            if 'silhouette_scores' not in results:
                results['silhouette_scores'] = {}
            results['silhouette_scores'][chunk_idx] = silhouette_scores
    
    # Summarize results across all chunks
    summary = {}
    for n_clusters in cluster_sizes:
        correct_chunks = sum(
            1 for chunk_idx in results['chunk_results'] 
            if n_clusters in results['chunk_results'][chunk_idx] and 
            results['chunk_results'][chunk_idx][n_clusters]['has_correct_answer']
        )
        total_chunks = sum(
            1 for chunk_idx in results['chunk_results'] 
            if n_clusters in results['chunk_results'][chunk_idx]
        )
        
        if total_chunks > 0:
            summary[n_clusters] = {
                'correct_chunks': correct_chunks,
                'total_chunks': total_chunks,
                'accuracy': correct_chunks / total_chunks
            }
    
    results['summary'] = summary
    
    # Calculate average silhouette scores across all chunks
    avg_silhouette_scores = {}
    if 'silhouette_scores' in results:
        for n_clusters in cluster_sizes:
            scores = [
                chunk_scores.get(n_clusters, 0) 
                for chunk_idx, chunk_scores in results['silhouette_scores'].items()
                if n_clusters in chunk_scores
            ]
            if scores:
                avg_silhouette_scores[n_clusters] = sum(scores) / len(scores)
    
    results['avg_silhouette_scores'] = avg_silhouette_scores
    
    return results

def generate_report(all_results, output_file=None):
    """Generate a simplified report focusing on correct answer presence by chunk and cluster size."""
    report = []
    
    # Overall statistics
    report.append("# Clustering Evaluation Report")
    report.append("")
    
    # Collect all chunk indices and cluster sizes across all problems
    all_chunk_indices = set()
    all_cluster_sizes = set()
    valid_results = [r for r in all_results if r]
    total_problems = len(valid_results)
    
    # Track which problems have correct answers for each chunk index and cluster size
    chunk_cluster_correct_problems = defaultdict(lambda: defaultdict(set))
    
    # Problem-specific details
    for result in valid_results:
        problem_id = result['problem_id']
        truth_answer = result['ground_truth']
        
        report.append(f"## Problem {problem_id}")
        report.append("")
        report.append(f"Ground truth answer: {truth_answer}")
        report.append("")
        
        # Find the best k based on silhouette score
        best_k = None
        if result.get('avg_silhouette_scores'):
            best_k_data = max(result['avg_silhouette_scores'].items(), key=lambda x: x[1])
            best_k = best_k_data[0]
            best_score = best_k_data[1]
            report.append(f"**Best cluster size based on silhouette score: {best_k} (score: {best_score:.4f})**")
            report.append("")
        
        # Create a table showing correct answer presence by chunk and cluster size
        # Get all chunk indices and cluster sizes
        chunk_indices = sorted(result['chunk_results'].keys())
        cluster_sizes = sorted({
            k for chunk_data in result['chunk_results'].values() 
            for k in chunk_data.keys()
        })
        
        # Update the global sets of chunk indices and cluster sizes
        all_chunk_indices.update(chunk_indices)
        all_cluster_sizes.update(cluster_sizes)
        
        if not chunk_indices or not cluster_sizes:
            report.append("No data available for this problem.")
            report.append("")
            continue
        
        # Create table header
        header = "| Chunk Index |"
        for size in cluster_sizes:
            header += f" {size} |"
        report.append(header)
        
        # Create separator row
        separator = "|-------------|"
        for _ in cluster_sizes:
            separator += "------|"
        report.append(separator)
        
        # Create table rows
        for chunk_idx in chunk_indices:
            row = f"| {chunk_idx} |"
            for size in cluster_sizes:
                # Check if this cluster size was evaluated for this chunk
                if size in result['chunk_results'][chunk_idx]:
                    has_correct = result['chunk_results'][chunk_idx][size]['has_correct_answer']
                    marker = "✓" if has_correct else "✗"
                    
                    # Track which problems have correct answers for each chunk-cluster combination
                    if has_correct:
                        chunk_cluster_correct_problems[chunk_idx][size].add(problem_id)
                else:
                    marker = "N/A"
                row += f" {marker} |"
            report.append(row)
        
        report.append("")
        
        # Add a summary row showing if any chunk has the correct answer for each cluster size
        summary_row = "| **Any Chunk** |"
        for size in cluster_sizes:
            any_correct = any(
                chunk_data.get(size, {}).get('has_correct_answer', False)
                for chunk_data in result['chunk_results'].values()
            )
            marker = "✓" if any_correct else "✗"
            summary_row += f" {marker} |"
        report.append(summary_row)
        
        report.append("")
        report.append("---")
        report.append("")
    
    # Add aggregate statistics section
    if total_problems > 0:
        report.append("## Aggregate Statistics")
        report.append("")
        report.append(f"Total problems analyzed: {total_problems}")
        report.append("")
        report.append("The following table shows, for each chunk index and cluster size combination, how many problems had at least one correct answer:")
        report.append("(Highlighted cells indicate maximum values for each row/column)")
        report.append("")
        
        # Sort the chunk indices and cluster sizes for consistent display
        all_chunk_indices = sorted(all_chunk_indices)
        all_cluster_sizes = sorted(all_cluster_sizes)
        
        # Calculate the maximum correctness for each chunk index (row)
        max_correctness_by_chunk = {}
        for chunk_idx in all_chunk_indices:
            max_correctness = 0
            for size in all_cluster_sizes:
                correct_count = len(chunk_cluster_correct_problems[chunk_idx][size])
                max_correctness = max(max_correctness, correct_count)
            max_correctness_by_chunk[chunk_idx] = max_correctness
        
        # Calculate the maximum correctness for each cluster size (column)
        max_correctness_by_size = {}
        for size in all_cluster_sizes:
            max_correctness = 0
            for chunk_idx in all_chunk_indices:
                correct_count = len(chunk_cluster_correct_problems[chunk_idx][size])
                max_correctness = max(max_correctness, correct_count)
            max_correctness_by_size[size] = max_correctness
        
        # Create table header
        header = "| Chunk Index |"
        for size in all_cluster_sizes:
            header += f" {size} |"
        report.append(header)
        
        # Create separator row
        separator = "|-------------|"
        for _ in all_cluster_sizes:
            separator += "------|"
        report.append(separator)
        
        # Create table rows
        for chunk_idx in all_chunk_indices:
            row = f"| {chunk_idx} |"
            for size in all_cluster_sizes:
                # Count how many problems had correct answers for this chunk-cluster combination
                correct_count = len(chunk_cluster_correct_problems[chunk_idx][size])
                ratio = f"{correct_count}/{total_problems}"
                percentage = f"{(correct_count/total_problems)*100:.1f}%"
                
                # Highlight maximum values
                is_max_for_row = correct_count == max_correctness_by_chunk[chunk_idx] and correct_count > 0
                is_max_for_col = correct_count == max_correctness_by_size[size] and correct_count > 0
                
                if is_max_for_row and is_max_for_col:
                    # Max for both row and column - bold and highlighted
                    cell = f" **{ratio}** ({percentage}) |"
                elif is_max_for_row:
                    # Max for row - bold
                    cell = f" **{ratio}** ({percentage}) |"
                elif is_max_for_col:
                    # Max for column - italics
                    cell = f" *{ratio}* ({percentage}) |"
                else:
                    cell = f" {ratio} ({percentage}) |"
                
                row += cell
            report.append(row)
        
        report.append("")
        
        # Add a summary row showing if any chunk has the correct answer for each cluster size
        summary_row = "| **Any Chunk** |"
        
        # Calculate the maximum correctness for the "Any Chunk" row
        problems_with_correct_by_size = {}
        max_any_chunk_correctness = 0
        
        for size in all_cluster_sizes:
            # Count problems where any chunk had a correct answer for this cluster size
            problems_with_correct = set()
            for chunk_idx in all_chunk_indices:
                problems_with_correct.update(chunk_cluster_correct_problems[chunk_idx][size])
            
            problems_with_correct_by_size[size] = problems_with_correct
            max_any_chunk_correctness = max(max_any_chunk_correctness, len(problems_with_correct))
        
        for size in all_cluster_sizes:
            problems_with_correct = problems_with_correct_by_size[size]
            correct_count = len(problems_with_correct)
            ratio = f"{correct_count}/{total_problems}"
            percentage = f"{(correct_count/total_problems)*100:.1f}%"
            
            # Highlight maximum values
            is_max = correct_count == max_any_chunk_correctness and correct_count > 0
            
            if is_max:
                # Max for the "Any Chunk" row - bold
                cell = f" **{ratio}** ({percentage}) |"
            else:
                cell = f" {ratio} ({percentage}) |"
            
            summary_row += cell
        
        report.append(summary_row)
        
        report.append("")
        report.append("**Bold** values indicate maximum correctness for each chunk index.")
        report.append("*Italic* values indicate maximum correctness for each cluster size.")
        report.append("**Bold** values in the 'Any Chunk' row indicate the best overall cluster sizes.")
        report.append("")
    
    report_text = "\n".join(report)
    
    # Write to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

def plot_accuracy_by_clusters(all_results, output_file=None):
    """Plot success rate vs. number of clusters."""
    # Calculate overall statistics across all problems
    overall_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_silhouette = defaultdict(list)
    problems_with_correct = defaultdict(set)  # Track which problems have at least one correct chunk
    
    total_problems = len([r for r in all_results if r])
    
    for result in all_results:
        if not result:
            continue
            
        problem_id = result['problem_id']
        
        for n_clusters, stats in result['summary'].items():
            overall_stats[n_clusters]['correct'] += stats['correct_chunks']
            overall_stats[n_clusters]['total'] += stats['total_chunks']
            
            # Check if any chunk for this problem has the correct answer
            if stats['correct_chunks'] > 0:
                problems_with_correct[n_clusters].add(problem_id)
        
        # Collect silhouette scores
        for n_clusters, score in result.get('avg_silhouette_scores', {}).items():
            overall_silhouette[n_clusters].append(score)
    
    # Calculate average silhouette scores across all problems
    avg_silhouette = {
        n_clusters: sum(scores) / len(scores) if scores else 0
        for n_clusters, scores in overall_silhouette.items()
    }
    
    # Calculate success rate for each cluster size (problems with at least one correct answer)
    cluster_sizes = sorted(problems_with_correct.keys())
    success_rates = [
        len(problems_with_correct[n]) / total_problems if total_problems > 0 else 0
        for n in cluster_sizes
    ]
    
    # Create plot with two y-axes
    fig, ax1 = plt.figure(figsize=(12, 7)), plt.gca()
    
    # Plot success rate on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Success Rate (% Problems with Correct Answer)', color=color)
    line1 = ax1.plot(cluster_sizes, success_rates, marker='o', color=color, label='Success Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.05])  # Set y-axis limits for percentage
    
    # Create a second y-axis for silhouette scores
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    
    # Plot silhouette scores if available
    if avg_silhouette:
        silhouette_sizes = sorted(avg_silhouette.keys())
        silhouette_scores = [avg_silhouette[n] for n in silhouette_sizes]
        line2 = ax2.plot(silhouette_sizes, silhouette_scores, marker='x', color=color, label='Silhouette Score')
        ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a title and legend
    plt.title('Success Rate and Silhouette Score vs. Number of Clusters')
    
    # Combine legends from both axes
    lines = line1
    labels = ['Success Rate']
    if avg_silhouette:
        lines += line2
        labels.append('Silhouette Score')
    
    ax1.legend(lines, labels, loc='best')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Cluster embeddings and evaluate effectiveness')
    parser.add_argument('--csv', type=str, default='AIME_2023_4.csv', help='Path to the CSV file with ground truth answers')
    parser.add_argument('--extracted', type=str, default='extracted_answers', help='Directory containing extracted answer files')
    parser.add_argument('--embeddings', type=str, default='chunked_embeddings', help='Directory containing chunked embeddings')
    parser.add_argument('--output', type=str, default='clustering_simplified_report.md', help='Output file for the evaluation report')
    parser.add_argument('--plot', type=str, default='clustering_accuracy.png', help='Output file for the accuracy plot')
    parser.add_argument('--cluster_sizes', type=str, default='10,15,20,25,30,35,40', help='Comma-separated list of cluster sizes to evaluate')
    parser.add_argument('--midway_only', action='store_true', help='Only process chunks up to the midway point')
    parser.add_argument('--max_problems', type=int, default=None, help='Maximum number of problems to process (default: all)')
    parser.add_argument('--specific_chunks', type=str, help='Comma-separated list of specific chunk indices to process (e.g., "0,1,2,3")')
    args = parser.parse_args()
    
    # Parse cluster sizes from comma-separated string
    cluster_sizes = [int(size.strip()) for size in args.cluster_sizes.split(',')]
    print(f"Evaluating cluster sizes: {cluster_sizes}")
    
    # Parse specific chunks if provided
    specific_chunks = None
    if args.specific_chunks:
        try:
            specific_chunks = [int(chunk) for chunk in args.specific_chunks.split(',')]
            print(f"Processing only specific chunks: {specific_chunks}")
        except ValueError:
            print(f"Error parsing specific_chunks '{args.specific_chunks}'. Processing all chunks.")
    
    # Load ground truth, extracted answers, and embeddings
    print("Loading ground truth answers...")
    ground_truth = load_ground_truth(args.csv)
    
    print("Loading extracted answers...")
    extracted_answers = load_extracted_answers(args.extracted)
    
    print("Loading embeddings...")
    embeddings_data = load_embeddings(args.embeddings)
    
    # Limit the number of problems if specified
    if args.max_problems is not None:
        print(f"Limiting to the first {args.max_problems} problems")
        problem_ids = sorted(list(embeddings_data.keys()))[:args.max_problems]
        embeddings_data = {pid: embeddings_data[pid] for pid in problem_ids}
    
    # Evaluate clustering for each problem
    all_results = []
    
    print(f"Evaluating clustering for {len(embeddings_data)} problems...")
    for problem_id, problem_data in tqdm(embeddings_data.items()):
        # Organize embeddings by chunk index
        chunk_embeddings, response_indices = organize_embeddings_by_chunk(problem_data)
        
        # Apply filtering based on midway_only or specific_chunks
        if args.midway_only or specific_chunks:
            original_chunk_count = len(chunk_embeddings)
            chunk_embeddings, response_indices = limit_to_midway(chunk_embeddings, response_indices, specific_chunks)
            print(f"Problem {problem_id}: Processing {len(chunk_embeddings)} chunks (filtered from {original_chunk_count})")
        
        # Evaluate clustering
        result = evaluate_clustering(
            problem_id, 
            chunk_embeddings, 
            response_indices, 
            extracted_answers, 
            ground_truth,
            cluster_sizes
        )
        
        all_results.append(result)
    
    # Generate and save report
    print("Generating report...")
    report = generate_report(all_results, args.output)
    
    # Plot accuracy by number of clusters
    print("Generating plot...")
    plot_accuracy_by_clusters(all_results, args.plot)
    
    print(f"Evaluation complete. Report saved to {args.output}")
    print(f"Accuracy plot saved to {args.plot}")

if __name__ == "__main__":
    main() 