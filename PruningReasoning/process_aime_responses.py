#!/usr/bin/env python3
import csv
import json
import os
import argparse
import time
import requests

def send_prompt_to_node_generate(prompt):
    """
    Send a prompt to the node_generate API and return the response.
    
    Args:
        prompt (str): The prompt to send to the API
        
    Returns:
        str: The response from the API
    """
    # API endpoint
    url = "https://p01--deepseek-r1-671b--9qglzwbr2nlh.code.run/v1/completions"
    
    # Request headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Request payload
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 23000,
        "top_p": 0.9,
        "n": 50  # Request 50 different completions
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the response data
        response_data = response.json()
        
        # No need to save the response here as we'll save it per question
        return response_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_aime_questions(csv_file):
    """
    Read AIME questions from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing AIME questions
        
    Returns:
        list: List of dictionaries with year, number, and question
    """
    questions = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to find the relevant columns
                year_col = next((col for col in row.keys() if 'year' in col.lower()), None)
                number_col = next((col for col in row.keys() if 'problem number' in col.lower() or 'problem' in col.lower()), None)
                question_col = next((col for col in row.keys() if 'question' in col.lower()), None)
                
                if year_col and number_col and question_col:
                    questions.append({
                        'year': row[year_col],
                        'number': row[number_col],
                        'question': row[question_col]
                    })
                print(questions)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    print(f"Read {len(questions)} questions from {csv_file}")
    return questions

def process_aime_question(question_data, output_dir="results"):
    """
    Process a single AIME question with the Deepseek model and save the result.
    
    Args:
        question_data (dict): Dictionary containing year, number, and question
        output_dir (str): Directory to save the result
    
    Returns:
        dict: Result dictionary with prompt and response
    """
    year = question_data['year']
    number = question_data['number']
    question = question_data['question']
    
    # Create a unique key for this question
    key = f"deepseek_{year}_{number}"
    
    try:
        # Send the question to the Deepseek model using node_generate
        print(f"Processing question {key}")
        response = send_prompt_to_node_generate(question)
        
        if response and 'choices' in response:
            # Store the result
            result = {
                "prompt": question,
                "responses": response['choices'],  # Store all 50 responses
                "year": year,
                "number": number
            }
            
            # Save individual result
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{key}_I.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"50 responses successfully saved to {output_file}")
            return result
        else:
            print(f"Error: No valid responses received for question {key}")
            return {
                "prompt": question,
                "error": "No valid responses received",
                "year": year,
                "number": number
            }
        
    except Exception as e:
        print(f"Error processing question {key}: {e}")
        return {
            "prompt": question,
            "error": str(e),
            "year": year,
            "number": number
        }

def process_all_aime_questions(csv_file, output_dir="results", delay=1):
    """
    Process all AIME questions from a CSV file and save the results.
    
    Args:
        csv_file (str): Path to the CSV file containing AIME questions
        output_dir (str): Directory to save the results
        delay (int): Delay between API calls in seconds
    
    Returns:
        dict: Dictionary of all results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read questions from CSV
    questions = read_aime_questions(csv_file)
    
    if not questions:
        print("No questions found. Please check the CSV file format.")
        return {}
    
    # Process each question
    results = {}
    
    for i, question_data in enumerate(questions):
        year = question_data['year']
        number = question_data['number']
        
        # Create a unique key for this question
        key = f"deepseek_{year}_{number}"
        print(f"Processing question {i+1}/{len(questions)}: {key}")
        
        # Process the question
        result = process_aime_question(question_data, output_dir)
        results[key] = result
        
    
    # Save all results to a single file
    all_results_file = os.path.join(output_dir, "all_deepseek_results_2023_I.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"All results saved to {all_results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Process AIME questions with the Deepseek model')
    parser.add_argument('--csv', type=str, default='AIME_2023_I.csv',
                        help='Path to the CSV file containing AIME questions')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save the results')
    parser.add_argument('--delay', type=int, default=1,
                        help='Delay between API calls in seconds')
    parser.add_argument('--question', type=str, default=None,
                        help='Process a specific question by year and number (format: 2023_4)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.question:
        # Process a specific question
        try:
            year, number = args.question.split('_')
            
            # Read all questions to find the matching one
            questions = read_aime_questions(args.csv)
            matching_question = None
            
            for q in questions:
                if q['year'] == year and q['number'] == number:
                    matching_question = q
                    break
            
            if matching_question:
                print(f"Processing AIME {year} Problem {number}...")
                process_aime_question(matching_question, args.output_dir)
            else:
                print(f"Question AIME {year} Problem {number} not found in {args.csv}")
        except ValueError:
            print("Invalid question format. Use format: 2023_4")
    else:
        # Process all questions
        print(f"Processing all AIME questions from {args.csv}...")
        process_all_aime_questions(args.csv, args.output_dir, args.delay)

if __name__ == "__main__":
    main() 