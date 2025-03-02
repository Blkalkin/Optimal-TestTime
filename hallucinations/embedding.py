import json
import numpy as np
import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import dotenv

dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Function to extract the answer part (after </think>)
def extract_answer(response_text):
    parts = response_text.split("</think>")
    if len(parts) > 1:
        return parts[1].strip()
    return response_text.strip()  # If no think tag, use full text

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Load and process the data
def analyze_continuation_divergence(file_path):
    # Group responses by original question
    question_groups = defaultdict(list)
    
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                original_id = data.get('original_question_id')
                response_text = data['response']['choices'][0]['text']
                
                # Extract the answer part (after thinking)
                answer = extract_answer(response_text)
                
                question_groups[original_id].append({
                    'question_id': data.get('question_id'),
                    'answer': answer
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
    
    # Analyze divergence for each question group
    results = {}
    
    for q_id, responses in question_groups.items():
        if len(responses) < 2:
            continue
            
        # Get embeddings for all answers
        embeddings = []
        for resp in responses:
            embedding = get_embedding(resp['answer'])
            embeddings.append(embedding)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Calculate average similarity (excluding self-comparisons)
        n = len(embeddings)
        total_sim = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_sim += similarities[i][j]
                count += 1
        
        avg_similarity = total_sim / count if count > 0 else 1.0
        divergence = 1.0 - avg_similarity
        
        results[q_id] = {
            'num_continuations': len(responses),
            'average_similarity': avg_similarity,
            'divergence': divergence,
            'similarity_matrix': similarities.tolist()
        }
    
    return results

# Visualization functions
def plot_divergence_results(results):
    # Plot average divergence across questions
    question_ids = list(results.keys())
    divergences = [results[q_id]['divergence'] for q_id in question_ids]
    
    plt.figure(figsize=(10, 6))
    plt.bar(question_ids, divergences)
    plt.title('Divergence Between Continuation Responses')
    plt.xlabel('Question ID')
    plt.ylabel('Divergence (1 - Avg Similarity)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_heatmap(results, question_id):
    # Create a heatmap for a specific question to visualize pairwise similarities
    if question_id not in results:
        print(f"Question ID {question_id} not found in results.")
        return
    
    similarities = np.array(results[question_id]['similarity_matrix'])
    n = similarities.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarities, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title(f'Pairwise Similarities for Question {question_id}')
    plt.xticks(range(n), [f"Resp {i+1}" for i in range(n)])
    plt.yticks(range(n), [f"Resp {i+1}" for i in range(n)])
    
    # Add text annotations in the cells
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{similarities[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if similarities[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    plt.show()

def save_results_to_csv(results, output_file):
    # Save summary results to CSV
    data = []
    for q_id, result in results.items():
        data.append({
            'question_id': q_id,
            'num_continuations': result['num_continuations'],
            'average_similarity': result['average_similarity'],
            'divergence': result['divergence']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # File path to the continuations responses
    file_path = "simpleqa_continuations_responses_test.jsonl"
    
    # Run the analysis
    print("Analyzing continuation divergence...")
    results = analyze_continuation_divergence(file_path)
    
    # Print summary statistics
    print("\nSummary of results:")
    print(f"Total questions analyzed: {len(results)}")
    
    if results:
        avg_divergence = sum(r['divergence'] for r in results.values()) / len(results)
        print(f"Average divergence across all questions: {avg_divergence:.4f}")
        
        # Find the question with highest divergence
        max_div_q = max(results.items(), key=lambda x: x[1]['divergence'])
        print(f"Question with highest divergence: {max_div_q[0]} (Divergence: {max_div_q[1]['divergence']:.4f})")
        
        # Find the question with lowest divergence
        min_div_q = min(results.items(), key=lambda x: x[1]['divergence'])
        print(f"Question with lowest divergence: {min_div_q[0]} (Divergence: {min_div_q[1]['divergence']:.4f})")
        
        # Plot the overall results
        plot_divergence_results(results)
        
        # Plot heatmap for the question with highest divergence
        plot_heatmap(results, max_div_q[0])
        
        # Save results to CSV
        save_results_to_csv(results, "continuations_test_divergence_results.csv")
    else:
        print("No results to analyze. Check if the file contains valid data.")

if __name__ == "__main__":
    main()
