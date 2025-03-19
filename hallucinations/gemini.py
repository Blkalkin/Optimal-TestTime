import json
import numpy as np
import re
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

# Function to extract response text from different possible structures
def get_response_text(response_obj):
    try:
        # Try to access as {"choices": [{"text": "..."}]}
        if "text" in response_obj["choices"][0]:
            return response_obj["choices"][0]["text"]
        # Try to access as {"choices": [{"message": {"content": "..."}}]}
        elif "message" in response_obj["choices"][0]:
            return response_obj["choices"][0]["message"]["content"]
        # Other potential formats could be handled here
        else:
            print(f"Unknown response format: {response_obj}")
            return ""
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting response text: {e}, Response: {response_obj}")
        return ""

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
                response_text = get_response_text(data['response'])
                
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
                response_text = get_response_text(data['response'])
                
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

def main():
    # File paths
    continuations_file = "simpleqa_continuations_responses_test_250.jsonl"
    original_file = "simpleqa_responses_test_250.jsonl"
    csv_results_file = "gemini_test_results_250.csv"
    
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
        
        # Save results to CSV if it doesn't exist yet
        if not os.path.exists(csv_results_file):
            save_results_to_csv(results, csv_results_file)
    else:
        print("No results to analyze. Check if the files contain valid data.")

if __name__ == "__main__":
    main()
