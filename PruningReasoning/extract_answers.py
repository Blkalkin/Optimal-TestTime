#!/usr/bin/env python3
import json
import os
import glob
import argparse
from openai import OpenAI

def extract_last_n_chars(text, n=500):
    """Extract the last n characters from a text string."""
    if len(text) <= n:
        return text
    return text[-n:]

def process_json_file(file_path, client, output_dir="extracted_answers", n=500):
    """Process a single JSON file, extracting answers from the last n characters of each response."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problem_id = f"{data.get('year', 'unknown')}_{data.get('number', 'unknown')}"
        prompt = data.get('prompt', '')
        responses = data.get('responses', [])
        
        results = []
        
        for i, response in enumerate(responses):
            text = response.get('text', '')
            last_n_chars = extract_last_n_chars(text, n)
            
            # Call GPT-4o to extract the answer
            gpt_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts the final numerical or algebraic answer from math solutions. Return ONLY a JSON object with the format {\"answer\": \"extracted_answer\"} where extracted_answer is the final answer found in the text. If no clear answer is found, return {\"answer\": null}."},
                    {"role": "user", "content": f"Extract the final answer from this text: {last_n_chars}"}
                ],
                response_format={"type": "json_object"}
            )
            
            extracted_answer = json.loads(gpt_response.choices[0].message.content)
            
            results.append({
                "response_index": i,
                "extracted_answer": extracted_answer.get("answer")
            })
        
        # Create output with problem info and extracted answers
        output = {
            "problem_id": problem_id,
            "prompt": prompt,
            "results": results
        }
        
        # Save to output file
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
            
        return output
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract answers from JSON files using GPT-4o')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--input_pattern', type=str, default='responses/*.json', help='Glob pattern for input files')
    parser.add_argument('--chars', type=int, default=500, help='Number of characters to extract from the end')
    parser.add_argument('--output_dir', type=str, default='extracted_answers', help='Directory to save extracted answers')
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Provide it via --api_key argument or OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=args.api_key)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving extracted answers to {args.output_dir}")
    
    # Process all matching files
    files = glob.glob(args.input_pattern)
    print(f"Found {len(files)} files to process")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        result = process_json_file(file_path, client, args.output_dir, args.chars)
        if result:
            print(f"Successfully processed {file_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()
