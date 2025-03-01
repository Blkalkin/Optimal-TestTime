import json
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import dotenv
import google.generativeai as genai
import os

dotenv.load_dotenv()

# Initialize Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract the answer part (after </think>)
def extract_answer(response_text):
    parts = response_text.split("</think>")
    if len(parts) > 1:
        return parts[1].strip()
    return response_text.strip()  # If no think tag, use full text

# Function to assess diversity using Gemini
def assess_diversity_with_gemini(answers, question):
    # Format the prompt for Gemini
    prompt = f"""
You are evaluating the diversity of answers to the same question.

QUESTION: {question}

Here are {len(answers)} different answers to analyze:

{chr(10).join([f"ANSWER {i+1}: {answer}" for i, answer in enumerate(answers)])}

TASK:
Analyze these answers for two types of diversity:
1. REASONING diversity: How diverse is the reasoning/thinking process across these answers? (scale: 0-10)
2. FINAL ANSWER diversity: How diverse are the final conclusions/answers? (scale: 0-10)

Note: 0 = almost identical, 10 = completely different approaches/conclusions

Respond with ONLY these two tags:
<REASONING_DIVERSITY>score</REASONING_DIVERSITY>
<FINAL_ANSWER_DIVERSITY>score</FINAL_ANSWER_DIVERSITY>

Do NOT include any explanations or additional text.
"""

    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    try:
        # Extract the reasoning diversity score
        reasoning_match = re.search(r'<REASONING_DIVERSITY>(\d+(?:\.\d+)?)</REASONING_DIVERSITY>', response_text)
        reasoning_diversity = float(reasoning_match.group(1)) if reasoning_match else 5.0
        
        # Extract the final answer diversity score
        final_match = re.search(r'<FINAL_ANSWER_DIVERSITY>(\d+(?:\.\d+)?)</FINAL_ANSWER_DIVERSITY>', response_text)
        final_diversity = float(final_match.group(1)) if final_match else 5.0
        
        return {
            "reasoning_diversity": reasoning_diversity,
            "final_diversity": final_diversity,
            "diversity_score": final_diversity  # Keep this for compatibility with existing code
        }
    except Exception as e:
        print(f"Warning: Could not parse diversity scores from Gemini response. Error: {e}")
        print(f"Raw response: {response_text}")
        
        # Fallback: try to extract any numbers
        scores = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
        if len(scores) >= 2:
            return {
                "reasoning_diversity": float(scores[0]),
                "final_diversity": float(scores[1]),
                "diversity_score": float(scores[1])  # Keep this for compatibility
            }
        
        return {
            "reasoning_diversity": 5.0,
            "final_diversity": 5.0,
            "diversity_score": 5.0  # Default value
        }

# Function to assess divergence from original answer using Gemini
def assess_original_divergence_with_gemini(original_answer, continuations, question):
    # Format the prompt for Gemini
    prompt = f"""
You are evaluating how much a set of continuation answers diverge from the original answer to a question.

QUESTION: {question}

ORIGINAL ANSWER: {original_answer}

Here are {len(continuations)} continuation answers to analyze:

{chr(10).join([f"CONTINUATION {i+1}: {cont}" for i, cont in enumerate(continuations)])}

TASK:
For each continuation, evaluate how much it diverges from the original answer in two ways:
1. REASONING divergence: How different is the reasoning/thinking approach? (scale: 0-10)
2. FINAL ANSWER divergence: How different is the conclusion/answer? (scale: 0-10)

Note: 0 = identical, 10 = completely different

Respond with ONLY these tags:
<REASONING_DIVERGENCE>comma-separated list of scores for each continuation</REASONING_DIVERGENCE>
<FINAL_ANSWER_DIVERGENCE>comma-separated list of scores for each continuation</FINAL_ANSWER_DIVERGENCE>

Do NOT include any explanations or additional text.
"""

    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    try:
        # Extract the reasoning divergence scores
        reasoning_match = re.search(r'<REASONING_DIVERGENCE>(.*?)</REASONING_DIVERGENCE>', response_text, re.DOTALL)
        reasoning_scores = []
        if reasoning_match:
            scores_text = reasoning_match.group(1).strip()
            reasoning_scores = [float(score.strip()) for score in scores_text.split(',') if score.strip()]
        
        # Extract the final answer divergence scores
        answer_match = re.search(r'<FINAL_ANSWER_DIVERGENCE>(.*?)</FINAL_ANSWER_DIVERGENCE>', response_text, re.DOTALL)
        answer_scores = []
        if answer_match:
            scores_text = answer_match.group(1).strip()
            answer_scores = [float(score.strip()) for score in scores_text.split(',') if score.strip()]
        
        # Calculate averages ourselves
        avg_reasoning = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0
        avg_answer = sum(answer_scores) / len(answer_scores) if answer_scores else 0
        
        return {
            'reasoning_divergence': avg_reasoning,
            'final_divergence': avg_answer,
            'divergence_score': avg_answer,  # For compatibility with existing code
            'individual_reasoning_scores': reasoning_scores,
            'individual_answer_scores': answer_scores
        }
        
    except Exception as e:
        print(f"Warning: Could not parse divergence scores from Gemini response. Error: {e}")
        print(f"Raw response: {response_text}")
        
        # Fallback: try to extract any numbers
        scores = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
        if len(scores) >= 2:
            # If we have at least 2 numbers, assume first set is reasoning, second is answer
            mid = len(scores) // 2
            reasoning_scores = [float(s) for s in scores[:mid]]
            answer_scores = [float(s) for s in scores[mid:]]
            
            avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)
            avg_answer = sum(answer_scores) / len(answer_scores)
            
            return {
                'reasoning_divergence': avg_reasoning,
                'final_divergence': avg_answer,
                'divergence_score': avg_answer,
                'individual_reasoning_scores': reasoning_scores,
                'individual_answer_scores': answer_scores
            }
        
        return {
            'reasoning_divergence': 0,
            'final_divergence': 0,
            'divergence_score': 0,
            'individual_reasoning_scores': [],
            'individual_answer_scores': []
        }

# Load original responses
def load_original_responses(file_path):
    original_answers = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                question_id = data.get('question_id')
                response_text = data['response']['choices'][0]['message']['content']
                
                # Extract the answer part (after thinking)
                answer = extract_answer(response_text)
                
                original_answers[question_id] = {
                    'question': data.get('question', ''),
                    'answer': answer
                }
            except (json.JSONDecodeError, KeyError) as e:
                # Skip lines that can't be parsed
                continue
    
    return original_answers

# Load and process the data
def analyze_continuation_diversity(continuations_file, original_file):
    # Load original responses
    print("Loading original responses...")
    original_responses = load_original_responses(original_file)
    
    # Group responses by original question
    print("Loading continuation responses...")
    question_groups = defaultdict(list)
    questions = {}
    
    with open(continuations_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                original_id = data.get('original_question_id')
                question_text = data.get('question')
                response_text = data['response']['choices'][0]['text']
                
                # Extract the answer part (after thinking)
                answer = extract_answer(response_text)
                
                question_groups[original_id].append({
                    'question_id': data.get('question_id'),
                    'answer': answer
                })
                
                # Store question text
                if original_id not in questions:
                    questions[original_id] = question_text
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
    
    # Analyze diversity for each question group
    results = {}
    
    for q_id, responses in question_groups.items():
        if len(responses) < 2:
            continue
        
        print(f"Analyzing question {q_id}...")
        
        # Extract just the answer texts
        answer_texts = [resp['answer'] for resp in responses]
        question_text = questions.get(q_id, f"Question {q_id}")
        
        # Get diversity assessment from Gemini
        print(f"- Assessing diversity for question {q_id}...")
        diversity_result = assess_diversity_with_gemini(answer_texts, question_text)
        
        # Normalize diversity score to 0-1 range for compatibility with visualization
        normalized_score = diversity_result.get('diversity_score', 0) / 10.0
        
        # Get divergence from original answer
        original_divergence_result = None
        if q_id in original_responses:
            print(f"- Assessing divergence from original for question {q_id}...")
            original_answer = original_responses[q_id]['answer']
            original_divergence_result = assess_original_divergence_with_gemini(
                original_answer, answer_texts, question_text
            )
        
        results[q_id] = {
            'question': question_text,
            'num_continuations': len(responses),
            'diversity_score': normalized_score,
            'diversity_details': diversity_result,
            'original_divergence_score': original_divergence_result.get('divergence_score', 0) / 10.0 if original_divergence_result else None,
            'original_divergence_details': original_divergence_result
        }
    
    return results

# Function to load results from CSV if it exists
def load_results_from_csv(csv_file):
    if not os.path.exists(csv_file):
        return None
    
    print(f"Loading existing results from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    results = {}
    for _, row in df.iterrows():
        q_id = row['question_id']
        results[q_id] = {
            'question': row['question_text'],
            'num_continuations': row['num_continuations'],
            'diversity_score': row['final_diversity'] / 10.0,  # Normalize to 0-1
            'diversity_details': {
                'reasoning_diversity': row['reasoning_diversity'],
                'final_diversity': row['final_diversity']
            },
            'original_divergence_score': row['final_divergence'] / 10.0 if pd.notna(row['final_divergence']) else None,
            'original_divergence_details': {
                'reasoning_divergence': row['reasoning_divergence'] if pd.notna(row['reasoning_divergence']) else None,
                'final_divergence': row['final_divergence'] if pd.notna(row['final_divergence']) else None
            }
        }
    
    return results

# Improved function to plot diversity results with focus on relationship between reasoning and final diversity
def plot_diversity_results(results):
    # Extract data
    question_ids = list(results.keys())
    reasoning_diversity = [results[q_id]['diversity_details']['reasoning_diversity'] for q_id in question_ids]
    final_diversity = [results[q_id]['diversity_details']['final_diversity'] for q_id in question_ids]
    
    # Sort by reasoning diversity score for better visualization
    sorted_indices = np.argsort(reasoning_diversity)
    question_ids = [question_ids[i] for i in sorted_indices]
    reasoning_diversity = [reasoning_diversity[i] for i in sorted_indices]
    final_diversity = [final_diversity[i] for i in sorted_indices]
    
    # Create figure and axis with better size
    plt.figure(figsize=(16, 8))
    
    # Create bar positions
    x = np.arange(len(question_ids))
    width = 0.35
    
    # Create bars with better colors
    plt.bar(x - width/2, reasoning_diversity, width, color='#3498db', label='Reasoning Diversity')
    plt.bar(x + width/2, final_diversity, width, color='#e74c3c', label='Final Answer Diversity')
    
    # Add labels and title with better font sizes
    plt.xlabel('Question ID', fontsize=12)
    plt.ylabel('Diversity Score (0-10)', fontsize=12)
    plt.title('Answer Diversity Across Questions', fontsize=16, fontweight='bold')
    plt.xticks(x, question_ids, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 10.5)  # Scale from 0 to 10 with a bit of margin
    
    # Add a horizontal line at the average with improved styling
    avg_reasoning = np.mean(reasoning_diversity)
    avg_final = np.mean(final_diversity)
    plt.axhline(y=avg_reasoning, color='#3498db', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Avg Reasoning: {avg_reasoning:.2f}')
    plt.axhline(y=avg_final, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Avg Final: {avg_final:.2f}')
    
    # Add text annotation for average values
    plt.text(x[-1]+1, avg_reasoning, f'{avg_reasoning:.2f}', fontsize=10, va='center', ha='left', color='#3498db')
    plt.text(x[-1]+1, avg_final, f'{avg_final:.2f}', fontsize=10, va='center', ha='left', color='#e74c3c')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('diversity_results.png', dpi=300)
    plt.close()
    
    # Enhanced scatter plot comparing reasoning vs final diversity
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with improved appearance
    scatter = plt.scatter(reasoning_diversity, final_diversity, alpha=0.7, s=80, 
                          c=np.arange(len(question_ids)), cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add question IDs as labels with better placement
    for i, q_id in enumerate(question_ids):
        plt.annotate(q_id, (reasoning_diversity[i], final_diversity[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Add diagonal line for reference
    plt.plot([0, 10], [0, 10], '--', color='gray', alpha=0.5, label='Equal Diversity Line')
    
    # Calculate and add regression line to show relationship
    z = np.polyfit(reasoning_diversity, final_diversity, 1)
    p = np.poly1d(z)
    plt.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100)), 'r-', 
             alpha=0.7, linewidth=2, label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Calculate and display correlation
    correlation = np.corrcoef(reasoning_diversity, final_diversity)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             alpha=0.2, facecolor='white'))
    
    # Add labels and title with better formatting
    plt.xlabel('Reasoning Diversity Score', fontsize=14)
    plt.ylabel('Final Answer Diversity Score', fontsize=14)
    plt.title('How Reasoning Diversity Affects Final Answer Diversity', fontsize=16, fontweight='bold')
    plt.xlim(0, 10.5)
    plt.ylim(0, 10.5)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Add text annotation explaining the relationship
    if correlation > 0.7:
        relationship_text = "Strong positive relationship: Higher reasoning diversity strongly predicts higher answer diversity"
    elif correlation > 0.4:
        relationship_text = "Moderate positive relationship: Higher reasoning diversity tends to predict higher answer diversity"
    elif correlation > 0.2:
        relationship_text = "Weak positive relationship: Higher reasoning diversity somewhat predicts higher answer diversity"
    elif correlation > -0.2:
        relationship_text = "No clear relationship: Reasoning diversity doesn't strongly predict answer diversity"
    else:
        relationship_text = "Negative relationship: Higher reasoning diversity associates with lower answer diversity"
    
    plt.figtext(0.5, 0.01, relationship_text, ha="center", fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", alpha=0.2, facecolor='white'))
    
    plt.savefig('diversity_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot divergence results
def plot_divergence_results(results):
    # Extract data for questions with valid divergence scores
    question_ids = []
    reasoning_divergence = []
    final_divergence = []
    
    for q_id, result in results.items():
        if result.get('original_divergence_details') is not None and result['original_divergence_details'].get('reasoning_divergence') is not None:
            question_ids.append(q_id)
            reasoning_divergence.append(result['original_divergence_details']['reasoning_divergence'])
            final_divergence.append(result['original_divergence_details']['final_divergence'])
    
    if not question_ids:
        print("No divergence data to plot")
        return
    
    # Sort by reasoning divergence score
    sorted_indices = np.argsort(reasoning_divergence)
    question_ids = [question_ids[i] for i in sorted_indices]
    reasoning_divergence = [reasoning_divergence[i] for i in sorted_indices]
    final_divergence = [final_divergence[i] for i in sorted_indices]
    
    # Create figure and axis with better dimensions
    plt.figure(figsize=(16, 8))
    
    # Create bar positions
    x = np.arange(len(question_ids))
    width = 0.35
    
    # Create bars with better colors
    plt.bar(x - width/2, reasoning_divergence, width, color='#9b59b6', alpha=0.8, label='Reasoning Divergence')
    plt.bar(x + width/2, final_divergence, width, color='#2ecc71', alpha=0.8, label='Final Answer Divergence')
    
    # Add labels and title with better font sizes
    plt.xlabel('Question ID', fontsize=12)
    plt.ylabel('Divergence from Original Answer (0-10)', fontsize=12)
    plt.title('How Much Continuations Diverge from Original Answers', fontsize=16, fontweight='bold')
    plt.xticks(x, question_ids, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 10.5)
    
    # Add horizontal lines at the averages with improved styling
    avg_reasoning = np.mean(reasoning_divergence)
    avg_final = np.mean(final_divergence)
    plt.axhline(y=avg_reasoning, color='#9b59b6', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Avg Reasoning: {avg_reasoning:.2f}')
    plt.axhline(y=avg_final, color='#2ecc71', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Avg Final: {avg_final:.2f}')
    
    # Add text annotation for average values
    plt.text(x[-1]+1, avg_reasoning, f'{avg_reasoning:.2f}', fontsize=10, va='center', ha='left', color='#9b59b6')
    plt.text(x[-1]+1, avg_final, f'{avg_final:.2f}', fontsize=10, va='center', ha='left', color='#2ecc71')
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('divergence_results.png', dpi=300)
    plt.close()
    
    # Enhanced scatter plot comparing reasoning vs final divergence
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with improved appearance
    scatter = plt.scatter(reasoning_divergence, final_divergence, alpha=0.7, s=80, 
                          c=np.arange(len(question_ids)), cmap='plasma', edgecolors='black', linewidth=1)
    
    # Add question IDs as labels with better placement
    for i, q_id in enumerate(question_ids):
        plt.annotate(q_id, (reasoning_divergence[i], final_divergence[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Add diagonal line for reference
    plt.plot([0, 10], [0, 10], '--', color='gray', alpha=0.5, label='Equal Divergence Line')
    
    # Calculate and add regression line
    z = np.polyfit(reasoning_divergence, final_divergence, 1)
    p = np.poly1d(z)
    plt.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100)), 'r-', 
             alpha=0.7, linewidth=2, label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Calculate and display correlation
    correlation = np.corrcoef(reasoning_divergence, final_divergence)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             alpha=0.2, facecolor='white'))
    
    # Add labels and title with better formatting
    plt.xlabel('Reasoning Divergence from Original', fontsize=14)
    plt.ylabel('Final Answer Divergence from Original', fontsize=14)
    plt.title('How Reasoning Divergence Affects Final Answer Divergence', fontsize=16, fontweight='bold')
    plt.xlim(0, 10.5)
    plt.ylim(0, 10.5)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('divergence_relationship.png', dpi=300)
    plt.close()

# Function to plot comparison between diversity and divergence
def plot_comparison(results):
    # Extract data for questions with all metrics
    question_ids = []
    reasoning_diversity = []
    final_diversity = []
    reasoning_divergence = []
    final_divergence = []
    
    for q_id, result in results.items():
        if result.get('original_divergence_details') is not None:
            question_ids.append(q_id)
            reasoning_diversity.append(result['diversity_details']['reasoning_diversity'])
            final_diversity.append(result['diversity_details']['final_diversity'])
            reasoning_divergence.append(result['original_divergence_details']['reasoning_divergence'])
            final_divergence.append(result['original_divergence_details']['final_divergence'])
    
    if not question_ids:
        print("No complete data to plot comparison")
        return
    
    # Create scatter plots for reasoning metrics
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Reasoning Diversity vs Reasoning Divergence
    axs[0, 0].scatter(reasoning_diversity, reasoning_divergence, alpha=0.7)
    for i, q_id in enumerate(question_ids):
        axs[0, 0].annotate(q_id, (reasoning_diversity[i], reasoning_divergence[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
    
    axs[0, 0].set_xlabel('Reasoning Diversity')
    axs[0, 0].set_ylabel('Reasoning Divergence from Original')
    axs[0, 0].set_title('Reasoning Diversity vs Reasoning Divergence')
    axs[0, 0].set_xlim(0, 10.5)
    axs[0, 0].set_ylim(0, 10.5)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    r_div_r_div_corr = np.corrcoef(reasoning_diversity, reasoning_divergence)[0, 1]
    axs[0, 0].text(0.05, 0.95, f'Correlation: {r_div_r_div_corr:.2f}', 
                 transform=axs[0, 0].transAxes, fontsize=12, 
                 verticalalignment='top')
    
    # Plot 2: Final Answer Diversity vs Final Answer Divergence
    axs[0, 1].scatter(final_diversity, final_divergence, alpha=0.7)
    for i, q_id in enumerate(question_ids):
        axs[0, 1].annotate(q_id, (final_diversity[i], final_divergence[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
    
    axs[0, 1].set_xlabel('Final Answer Diversity')
    axs[0, 1].set_ylabel('Final Answer Divergence from Original')
    axs[0, 1].set_title('Final Answer Diversity vs Final Answer Divergence')
    axs[0, 1].set_xlim(0, 10.5)
    axs[0, 1].set_ylim(0, 10.5)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    f_div_f_div_corr = np.corrcoef(final_diversity, final_divergence)[0, 1]
    axs[0, 1].text(0.05, 0.95, f'Correlation: {f_div_f_div_corr:.2f}', 
                 transform=axs[0, 1].transAxes, fontsize=12, 
                 verticalalignment='top')
    
    # Plot 3: Reasoning Diversity vs Final Answer Divergence
    axs[1, 0].scatter(reasoning_diversity, final_divergence, alpha=0.7)
    for i, q_id in enumerate(question_ids):
        axs[1, 0].annotate(q_id, (reasoning_diversity[i], final_divergence[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
    
    axs[1, 0].set_xlabel('Reasoning Diversity')
    axs[1, 0].set_ylabel('Final Answer Divergence from Original')
    axs[1, 0].set_title('Reasoning Diversity vs Final Answer Divergence')
    axs[1, 0].set_xlim(0, 10.5)
    axs[1, 0].set_ylim(0, 10.5)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    r_div_f_div_corr = np.corrcoef(reasoning_diversity, final_divergence)[0, 1]
    axs[1, 0].text(0.05, 0.95, f'Correlation: {r_div_f_div_corr:.2f}', 
                 transform=axs[1, 0].transAxes, fontsize=12, 
                 verticalalignment='top')
    
    # Plot 4: Final Answer Diversity vs Reasoning Divergence
    axs[1, 1].scatter(final_diversity, reasoning_divergence, alpha=0.7)
    for i, q_id in enumerate(question_ids):
        axs[1, 1].annotate(q_id, (final_diversity[i], reasoning_divergence[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
    
    axs[1, 1].set_xlabel('Final Answer Diversity')
    axs[1, 1].set_ylabel('Reasoning Divergence from Original')
    axs[1, 1].set_title('Final Answer Diversity vs Reasoning Divergence')
    axs[1, 1].set_xlim(0, 10.5)
    axs[1, 1].set_ylim(0, 10.5)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    f_div_r_div_corr = np.corrcoef(final_diversity, reasoning_divergence)[0, 1]
    axs[1, 1].text(0.05, 0.95, f'Correlation: {f_div_r_div_corr:.2f}', 
                 transform=axs[1, 1].transAxes, fontsize=12, 
                 verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('diversity_divergence_matrix.png')
    plt.close()

# Function to save results to CSV
def save_results_to_csv(results, output_file):
    # Prepare data for DataFrame
    data = []
    
    for q_id, result in results.items():
        row = {
            'question_id': q_id,
            'question_text': result.get('question', ''),
            'num_continuations': result.get('num_continuations', 0),
            'reasoning_diversity': result.get('diversity_details', {}).get('reasoning_diversity', 'N/A'),
            'final_diversity': result.get('diversity_details', {}).get('final_diversity', 'N/A'),
            'reasoning_divergence': result.get('original_divergence_details', {}).get('reasoning_divergence', 'N/A'),
            'final_divergence': result.get('original_divergence_details', {}).get('final_divergence', 'N/A')
        }
        data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Create an additional visualization specifically focused on reasoning vs answer diversity
def plot_reasoning_affects_answer(results):
    # Extract data
    question_ids = list(results.keys())
    reasoning_diversity = [results[q_id]['diversity_details']['reasoning_diversity'] for q_id in question_ids]
    final_diversity = [results[q_id]['diversity_details']['final_diversity'] for q_id in question_ids]
    
    num_continuations = [results[q_id]['num_continuations'] for q_id in question_ids]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot with bubble size representing number of continuations
    sizes = [30 * n for n in num_continuations]  # Scale the sizes
    scatter = plt.scatter(reasoning_diversity, final_diversity, s=sizes, alpha=0.6, 
                          c=np.array(final_diversity) - np.array(reasoning_diversity), 
                          cmap='coolwarm', edgecolors='black', linewidth=1)
    
    # Add colorbar to show the difference between final and reasoning diversity
    cbar = plt.colorbar(scatter)
    cbar.set_label('Final Answer Diversity - Reasoning Diversity', fontsize=12)
    
    # Calculate quadrants
    avg_reasoning = np.mean(reasoning_diversity)
    avg_final = np.mean(final_diversity)
    
    # Add quadrant lines
    plt.axvline(x=avg_reasoning, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=avg_final, color='gray', linestyle='--', alpha=0.5)
    
    # Label the quadrants
    plt.text(1, 9.5, 'High Answer Diversity\nLow Reasoning Diversity', 
             fontsize=10, ha='left', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1, facecolor='yellow'))
    
    plt.text(9, 9.5, 'High Answer Diversity\nHigh Reasoning Diversity', 
             fontsize=10, ha='right', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1, facecolor='green'))
    
    plt.text(1, 0.5, 'Low Answer Diversity\nLow Reasoning Diversity', 
             fontsize=10, ha='left', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1, facecolor='red'))
    
    plt.text(9, 0.5, 'Low Answer Diversity\nHigh Reasoning Diversity', 
             fontsize=10, ha='right', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", alpha=0.1, facecolor='orange'))
    
    # Add labels for each point
    for i, q_id in enumerate(question_ids):
        plt.annotate(q_id, (reasoning_diversity[i], final_diversity[i]), 
                   textcoords="offset points", xytext=(0,7), ha='center', fontsize=9)
    
    # Add diagonal line (where reasoning diversity = final diversity)
    plt.plot([0, 10], [0, 10], '-', color='black', alpha=0.5, 
             label='Equal Diversity Line (Reasoning = Final)')
    
    # Calculate and add regression line
    z = np.polyfit(reasoning_diversity, final_diversity, 1)
    p = np.poly1d(z)
    plt.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100)), 'r-', 
             alpha=0.7, linewidth=2, label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Calculate correlation and other statistics
    correlation = np.corrcoef(reasoning_diversity, final_diversity)[0, 1]
    
    # Count points in each quadrant
    q1 = sum(1 for i in range(len(reasoning_diversity)) 
             if reasoning_diversity[i] > avg_reasoning and final_diversity[i] > avg_final)
    q2 = sum(1 for i in range(len(reasoning_diversity)) 
             if reasoning_diversity[i] <= avg_reasoning and final_diversity[i] > avg_final)
    q3 = sum(1 for i in range(len(reasoning_diversity)) 
             if reasoning_diversity[i] <= avg_reasoning and final_diversity[i] <= avg_final)
    q4 = sum(1 for i in range(len(reasoning_diversity)) 
             if reasoning_diversity[i] > avg_reasoning and final_diversity[i] <= avg_final)
    
    # Add text box with statistics
    stats_text = (
        f"Correlation: {correlation:.2f}\n"
        f"Regression: Final = {z[0]:.2f} × Reasoning + {z[1]:.2f}\n"
        f"Average Reasoning Diversity: {avg_reasoning:.2f}\n"
        f"Average Final Diversity: {avg_final:.2f}\n\n"
        f"Points above diagonal (Final > Reasoning): {sum(1 for i in range(len(reasoning_diversity)) if final_diversity[i] > reasoning_diversity[i])}\n"
        f"Points below diagonal (Final < Reasoning): {sum(1 for i in range(len(reasoning_diversity)) if final_diversity[i] < reasoning_diversity[i])}"
    )
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", alpha=0.2, facecolor='white'))
    
    # Add labels and title
    plt.xlabel('Reasoning Diversity', fontsize=14)
    plt.ylabel('Final Answer Diversity', fontsize=14)
    plt.title('How Reasoning Diversity Affects Final Answer Diversity', fontsize=16, fontweight='bold')
    plt.xlim(0, 10.5)
    plt.ylim(0, 10.5)
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend(fontsize=11, loc='lower right')
    
    # Add summary interpretation
    if correlation > 0.7:
        interpretation = "Strong Relationship: Reasoning diversity is a strong predictor of final answer diversity"
    elif correlation > 0.4:
        interpretation = "Moderate Relationship: Reasoning diversity moderately predicts final answer diversity"
    else:
        interpretation = "Weak Relationship: Reasoning diversity weakly predicts final answer diversity"
        
    plt.figtext(0.5, 0.01, interpretation, ha="center", fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", alpha=0.2, facecolor='white'))
    
    plt.tight_layout()
    plt.savefig('reasoning_affects_answer.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    continuations_file = "simpleqa_continuations_responses_500.jsonl"
    original_file = "simpleqa_responses_500.jsonl"
    csv_results_file = "gemini_500_results.csv"
    
    # First try to load results from CSV if it exists
    results = load_results_from_csv(csv_results_file)
    
    # If no CSV exists, run the analysis
    if results is None:
        print("Analyzing continuation diversity and divergence using Gemini...")
        results = analyze_continuation_diversity(continuations_file, original_file)
    
    # Print summary statistics
    print("\nSummary of results:")
    print(f"Total questions analyzed: {len(results)}")
    
    if results:
        # Calculate diversity statistics
        avg_diversity = sum(r['diversity_score'] for r in results.values()) / len(results)
        print(f"Average diversity across all questions: {avg_diversity:.4f}")
        
        # Find the question with highest diversity
        max_div_q = max(results.items(), key=lambda x: x[1]['diversity_score'])
        print(f"Question with highest diversity: {max_div_q[0]} (Score: {max_div_q[1]['diversity_score']:.4f})")
        
        # Find the question with lowest diversity
        min_div_q = min(results.items(), key=lambda x: x[1]['diversity_score'])
        print(f"Question with lowest diversity: {min_div_q[0]} (Score: {min_div_q[1]['diversity_score']:.4f})")
        
        # Calculate divergence statistics
        divergence_results = {q_id: r for q_id, r in results.items() if r['original_divergence_score'] is not None}
        if divergence_results:
            avg_divergence = sum(r['original_divergence_score'] for r in divergence_results.values()) / len(divergence_results)
            print(f"\nAverage divergence from original answers: {avg_divergence:.4f}")
            
            # Find the question with highest divergence
            max_orig_div_q = max(divergence_results.items(), key=lambda x: x[1]['original_divergence_score'])
            print(f"Question with highest divergence from original: {max_orig_div_q[0]} (Score: {max_orig_div_q[1]['original_divergence_score']:.4f})")
            
            # Find the question with lowest divergence
            min_orig_div_q = min(divergence_results.items(), key=lambda x: x[1]['original_divergence_score'])
            print(f"Question with lowest divergence from original: {min_orig_div_q[0]} (Score: {min_orig_div_q[1]['original_divergence_score']:.4f})")
        
        # Plot the enhanced visualizations
        plot_diversity_results(results)
        
        # Create the new specialized visualization
        plot_reasoning_affects_answer(results)
        
        if divergence_results:
            plot_divergence_results(results)
            plot_comparison(results)
        
        # Save results to CSV if it doesn't exist yet
        if not os.path.exists(csv_results_file):
            save_results_to_csv(results, csv_results_file)
    else:
        print("No results to analyze. Check if the files contain valid data.")

if __name__ == "__main__":
    main()
