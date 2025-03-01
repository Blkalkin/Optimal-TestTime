#!/usr/bin/env python3
"""
SimpleQA evaluation script using DeepSeek for answering and GPT-4o for grading.
"""

import pandas as pd
import argparse
import random
import re
import os
import json
from datetime import datetime
import requests
import dotenv
from tqdm import tqdm
import openai

# Import from simple-evals
from simple_evals.simpleqa_eval import GRADER_TEMPLATE, CHOICE_LETTER_TO_STRING
from simple_evals.sampler.chat_completion_sampler import ChatCompletionSampler

dotenv.load_dotenv()

def query_fireworks_api(prompt, api_key=None, thinking_continuation=None):
    """Send a prompt to the Fireworks API and return the response."""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    # Prepare the message
    payload = {
            "model": "accounts/fireworks/models/deepseek-r1",
            "max_tokens": 2000,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    
    if thinking_continuation and thinking_continuation.strip():
        # If we have previous thinking, include it in the prompt
        url = "https://api.fireworks.ai/inference/v1/completions"
        payload = {
            "model": "accounts/fireworks/models/deepseek-r1",
            "max_tokens": 20480,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.7,
            "prompt": f"{prompt}\n\n<think>{thinking_continuation}"
        }

    #print(payload)
    
    # Get API key from environment variable if not provided
    if not api_key:
        api_key = os.environ.get("FIREWORKS_API_KEY", "<API_KEY>")
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    #print(response.json())
    return response.json()

def save_to_jsonl(data, filename="responses.jsonl"):
    """Save data to a JSONL file, appending if the file exists."""
    # Add timestamp to the data
    data["timestamp"] = datetime.now().isoformat()
    
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")
    
    print(f"Response saved to {filename}")

def extract_thinking(question_id, filename="responses.jsonl"):
    """Extract the latest thinking for a given question ID from the JSONL file."""
    if not os.path.exists(filename):
        return None
    
    latest_entry = None
    
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("question_id") == question_id:
                latest_entry = data
    
    thinking = None
    if latest_entry and "response" in latest_entry and "choices" in latest_entry["response"]:
        if len(latest_entry["response"]["choices"]) > 0:
            content = latest_entry["response"]["choices"][0]["message"]["content"]
            
            # Extract thinking part between <think> and </think> tags
            thinking_pattern = r'<think>(.*?)</think>'
            thinking_match = re.search(thinking_pattern, content, re.DOTALL)
            
            if thinking_match:
                thinking = thinking_match.group(1).strip()
    
    return thinking

def generate_thinking_continuations(input_jsonl="responses.jsonl", output_jsonl="continuations.jsonl", num_continuations=3):
    """
    Generate thinking continuations for each question in the input JSONL file.
    Creates a new JSONL file with follow-up question IDs (e.g., 1.1, 1.2, 1.3).
    """
    if not os.path.exists(input_jsonl):
        print(f"Input file {input_jsonl} does not exist.")
        return
    
    # Read all entries from the input file
    entries = []
    with open(input_jsonl, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    # Group entries by question ID
    question_groups = {}
    for entry in entries:
        q_id = entry.get("question_id")
        if q_id and "response" in entry:
            if q_id not in question_groups:
                question_groups[q_id] = []
            question_groups[q_id].append(entry)
    
    # Create continuations for each question
    continuations = []
    for q_id, group in question_groups.items():
        # Sort entries by timestamp if available
        if "timestamp" in group[0]:
            group.sort(key=lambda x: x.get("timestamp", ""))
        
        # Get the latest entry
        latest_entry = group[-1]
        original_question = latest_entry["question"]
        
        # Extract thinking from the latest entry
        thinking = extract_thinking(q_id, input_jsonl)
        
        if thinking:
            print(f"Found thinking for question ID {q_id}: {thinking[:50]}...")
            # Generate continuation questions
            for i in range(1, num_continuations + 1):
                new_id = f"{q_id}.{i}"
                continuation = {
                    "question_id": new_id,
                    "original_question_id": q_id,
                    "question": original_question,
                    "thinking_continuation": thinking,
                    "timestamp": datetime.now().isoformat()
                }
                # if i < num_continuations/2:
                continuation["thinking_continuation"] = continuation["thinking_continuation"] + " No, but"
                # else:
                #     continuation["thinking_continuation"] = continuation["thinking_continuation"] + " Yes, and"
                continuations.append(continuation)
                print(f"Generated continuation with ID {new_id}")
        else:
            print(f"No thinking found for question ID {q_id}")
    
    # Write continuations to the output file
    with open(output_jsonl, "w") as f:
        for continuation in continuations:
            f.write(json.dumps(continuation) + "\n")
    
    print(f"Generated {len(continuations)} thinking continuations in {output_jsonl}")
    return continuations

def process_continuations(continuations_jsonl="continuations.jsonl", output_jsonl="continuation_responses.jsonl", api_key=None):
    """
    Process a JSONL file of thinking continuations and generate responses.
    """
    if not os.path.exists(continuations_jsonl):
        print(f"Continuations file {continuations_jsonl} does not exist.")
        return
    
    # Get API key
    if not api_key:
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            print("Warning: FIREWORKS_API_KEY environment variable not set.")
            api_key = input("Please enter your Fireworks API key: ")
    
    # Read all continuations
    continuations = []
    with open(continuations_jsonl, "r") as f:
        for line in f:
            continuations.append(json.loads(line.strip()))

    # Process each continuation
    for continuation in tqdm(continuations):
        question_id = continuation["question_id"]
        original_id = continuation.get("original_question_id", "unknown")
        question_text = continuation["question"]
        thinking = continuation.get("thinking_continuation")
        
        print(f"Processing continuation {question_id} (from {original_id})")
        
        if thinking:
            print(f"Continuing from previous thinking")
        else:
            print(f"No thinking continuation found")
        
        # Query API with prompt and thinking continuation
        response = query_fireworks_api(question_text, api_key, thinking)
        
        # Print the response content
        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["text"]
            print(f"Answer: {answer[:200]}...")  # Print first 200 chars of answer
        else:
            print(f"Error or unexpected response format: {response}")
        
        # Save the full response to the output JSONL
        save_to_jsonl({
            "question_id": question_id,
            "original_question_id": original_id,
            "question": question_text,
            "response": response,
            "thinking_continuation": thinking is not None
        }, filename=output_jsonl)
    
    print(f"All {len(continuations)} continuations have been processed and saved to {output_jsonl}")

def generate_initial_answers(num_examples=None, output_jsonl="simpleqa_responses.jsonl", api_key=None):
    """
    Generate initial answers to SimpleQA questions.
    
    Args:
        num_examples: Number of examples to process (None for all)
        output_jsonl: File to save the original responses
        api_key: DeepSeek API key
    """
    # Get API key
    if not api_key:
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            print("Warning: FIREWORKS_API_KEY environment variable not set.")
            api_key = input("Please enter your Fireworks API key: ")
    
    # Load SimpleQA dataset
    print(f"Loading SimpleQA dataset...")
    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]
    
    # Sample examples if requested
    if num_examples and num_examples < len(examples):
        examples = random.sample(examples, min(num_examples, len(examples)))
    
    print(f"Generating answers for {len(examples)} SimpleQA examples")
    
    # Clear output file
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)
        print(f"Removed existing file: {output_jsonl}")
    
    # Process each example
    for i, example in enumerate(tqdm(examples)):
        question_id = str(i + 1)
        question_text = example.get("problem", "")
        
        print(f"Processing question {question_id}: {question_text[:100]}...")
        
        # Query API with the question
        response = query_fireworks_api(question_text, api_key)
        
        # Print the response content
        if "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0]["message"]["content"]
            print(f"Answer: {answer[:200]}...")  # Print first 200 chars of answer
        else:
            print(f"Error or unexpected response format: {response}")
        
        # Save the full response to JSONL
        save_to_jsonl({
            "question_id": question_id,
            "question": question_text,
            "response": response,
            "gold_answer": example.get("answer", ""),
            "continued_thinking": False
        }, filename=output_jsonl)
    
    print(f"All initial answers have been generated and saved to {output_jsonl}")
    return output_jsonl

def get_consensus(continuation_responses_jsonl="simpleqa_continuations_responses.jsonl"):
    """
    Get the consensus answer for each question from the continuation responses.
    Uses GPT-4o to determine if there's a clear consensus among answers.
    """
    if not os.path.exists(continuation_responses_jsonl):
        print(f"Continuation responses file {continuation_responses_jsonl} does not exist.")
        return
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Dictionary to store answers and questions for each original question
    question_data = {}
    
    # Parse the continuation responses
    with open(continuation_responses_jsonl, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            original_question_id = entry.get("original_question_id", "")
            question_text = entry.get("question", "")
            
            # Get the content from the response
            if "response" in entry and "choices" in entry["response"] and len(entry["response"]["choices"]) > 0:
                content = entry["response"]["choices"][0].get("text", "")
                
                # Remove anything before the </think> tag if it exists
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # Initialize the question data structure if needed
                if original_question_id not in question_data:
                    question_data[original_question_id] = {
                        "question_text": question_text,
                        "answers": []
                    }
                
                # Add to the answers list for this question
                question_data[original_question_id]["answers"].append(content)
    
    # Get consensus for each question using GPT-4o
    consensus_results = {}
    for question_id, data in question_data.items():
        question_text = data["question_text"]
        answers = data["answers"]
        
        print(f"Processing question {question_id}: {question_text[:100]}...")
        
        # Step 1: Extract core answers from each verbose answer
        core_answers = []
        for answer in answers:
            extract_prompt = f"""
            Given this question: "{question_text}"
            
            And this answer: "{answer}"
            
            Extract ONLY the core answer phrase that directly addresses the question.
            Return just the essential answer without any explanation or reasoning.
            """
            
            extract_response = client.chat.completions.create(
                model="gpt-4o",  # Using full GPT-4o instead of mini
                messages=[{"role": "user", "content": extract_prompt}]
            )
            
            core_answer = extract_response.choices[0].message.content.strip()
            core_answers.append(core_answer)
        
        # Step 2: Determine consensus directly with GPT-4o
        consensus_prompt = f"""
        I have multiple answers to the question: "{question_text}"
        
        Here are the answers:
        {core_answers}
        
        First, identify distinct answer options and count their frequencies.
        
        Then, determine if there's a clear consensus among these answers. 
        A clear consensus exists when:
        1. One answer has significantly more occurrences than others (over 50% of all answers)
        2. The multiple similar answers clearly converge on the same information
        
        If there's a clear consensus, return:
        {{"consensus": true, "answer": "[the consensus answer]"}}
        
        If there's no clear consensus or significant disagreement between answers, return:
        {{"consensus": false, "answer": "I don't know."}}
        
        Please output ONLY the JSON object with your determination.
        """
        
        consensus_response = client.chat.completions.create(
            model="gpt-4o",  # Using full GPT-4o instead of mini
            messages=[{"role": "user", "content": consensus_prompt}],
            response_format={"type": "json_object"}
        )
        
        consensus_json = consensus_response.choices[0].message.content
        
        consensus_results[question_id] = {
            "question": question_text,
            "consensus": consensus_json
        }
        
        print(f"Answers for question {question_id}: {core_answers}")
        print(f"Consensus: {consensus_results[question_id]['consensus']}")
    
    return consensus_results

def update_responses_with_consensus(responses_jsonl="simpleqa_responses.jsonl", output_jsonl="simpleqa_consensus_responses.jsonl", consensus_results=None):
    """
    Update the responses with the consensus answer.
    If GPT-4o determines there's no clear consensus, use "I don't know."
    """
    if not os.path.exists(responses_jsonl):
        print(f"Responses file {responses_jsonl} does not exist.")
        return
    
    # Get consensus results if not provided
    if consensus_results is None:
        print("Generating consensus results...")
        consensus_results = get_consensus()
    
    # Read all entries from the responses file
    entries = []
    with open(responses_jsonl, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    # Prepare output entries
    output_entries = []
    
    for entry in entries:
        question_id = entry.get("question_id", "unknown")
        question = entry.get("question", "")
        gold_answer = entry.get("gold_answer", "")
        
        # Get original response
        original_answer = ""
        if "response" in entry and "choices" in entry["response"] and len(entry["response"]["choices"]) > 0:
            original_answer = entry["response"]["choices"][0]["message"]["content"]
            # Remove thinking part if present
            if "</think>" in original_answer:
                original_answer = original_answer.split("</think>", 1)[1].strip()
        
        # Check if we have consensus results for this question
        consensus_answer = original_answer  # Default to original answer
        
        if question_id in consensus_results:
            # Parse the consensus JSON
            try:
                consensus_data = consensus_results[question_id]["consensus"]
                
                # Parse as JSON
                consensus_obj = json.loads(consensus_data)
                
                # Use the determined answer directly from GPT-4o
                if consensus_obj.get("consensus", False):
                    consensus_answer = consensus_obj.get("answer", original_answer)
                else:
                    consensus_answer = "I don't know."
                    
            except Exception as e:
                print(f"Error parsing consensus for question {question_id}: {e}")
        
        # Create output entry
        output_entry = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": consensus_answer
        }
        
        output_entries.append(output_entry)
        
        print(f"Question {question_id}: {'Using consensus' if consensus_answer != original_answer else 'Using original answer'}")
    
    # Write output entries to file
    with open(output_jsonl, "w") as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Updated {len(output_entries)} entries with consensus answers and saved to {output_jsonl}")
    return output_jsonl

def evaluate_simpleqa_answers(eval_jsonl="simpleqa_eval.jsonl"):
    """
    Evaluate SimpleQA answers using GPT-4o as the grader.
    
    Args:
        eval_jsonl: JSONL file containing questions, gold answers, and predicted answers in format:
                    {"question_id": "...", "question": "...", "gold_answer": "...", "predicted_answer": "..."}
    """
    if not os.path.exists(eval_jsonl):
        print(f"Evaluation file {eval_jsonl} does not exist.")
        return
    
    # Initialize grader model
    grader_model = ChatCompletionSampler(model="gpt-4o")
    
    # Read all entries
    eval_entries = []
    with open(eval_jsonl, "r") as f:
        for line in f:
            eval_entries.append(json.loads(line.strip()))
    
    # Results tracking
    results = {
        "CORRECT": 0,
        "INCORRECT": 0,
        "NOT_ATTEMPTED": 0,
    }
    
    # Process and grade each entry
    print(f"Evaluating {len(eval_entries)} responses...")
    for entry in tqdm(eval_entries):
        question_id = entry.get("question_id", "unknown")
        question_text = entry.get("question", "")
        gold_answer = entry.get("gold_answer", "")
        predicted_answer = entry.get("predicted_answer", "")
        
        # Skip if no gold answer or predicted answer
        if not gold_answer:
            tqdm.write(f"Skipping question {question_id}: No gold answer")
            continue
        
        # Grade the answer
        try:
            grader_prompt = GRADER_TEMPLATE.format(
                question=question_text,
                target=gold_answer,
                predicted_answer=predicted_answer,
            )
            
            prompt_messages = [
                grader_model._pack_message(content=grader_prompt, role="user")
            ]
            grading_response = grader_model(prompt_messages)
            
            match = re.search(r"(A|B|C)", grading_response)
            grade_letter = match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
            grade_string = CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")
            
            # Update results
            results[grade_string] += 1
            
            tqdm.write(f"Question {question_id}: Graded as {grade_string}")
            
        except Exception as e:
            tqdm.write(f"Error grading question {question_id}: {e}")
            results["NOT_ATTEMPTED"] += 1
    
    # Calculate metrics
    total = sum(results.values())
    is_correct = results["CORRECT"] / total if total > 0 else 0
    is_incorrect = results["INCORRECT"] / total if total > 0 else 0
    is_not_attempted = results["NOT_ATTEMPTED"] / total if total > 0 else 0
    is_given_attempted = is_correct + is_incorrect
    
    accuracy_given_attempted = is_correct / is_given_attempted if is_given_attempted > 0 else 0
    f1 = (2 * accuracy_given_attempted * is_correct) / (accuracy_given_attempted + is_correct) if (accuracy_given_attempted + is_correct) > 0 else 0
    
    # Print final results
    print("\n========== RESULTS ==========")
    print(f"Examples evaluated: {total}")
    print(f"Correct: {results['CORRECT']} ({is_correct:.2%})")
    print(f"Incorrect: {results['INCORRECT']} ({is_incorrect:.2%})")
    print(f"Not Attempted: {results['NOT_ATTEMPTED']} ({is_not_attempted:.2%})")
    print(f"Accuracy Given Attempted: {accuracy_given_attempted:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=============================")
    
    return {
        "total": total,
        "correct": results["CORRECT"],
        "incorrect": results["INCORRECT"],
        "not_attempted": results["NOT_ATTEMPTED"],
        "accuracy_given_attempted": accuracy_given_attempted,
        "f1": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate answers to SimpleQA questions")
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "continuations", "process", "consensus", "evaluate"], 
                        help="Mode: generate initial answers, create continuations, process continuations, or evaluate answers")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to process")
    parser.add_argument("--num-continuations", type=int, default=3, help="Number of continuations per question")
    parser.add_argument("--responses-file", type=str, default="simpleqa_responses.jsonl", help="File to save/load responses")
    parser.add_argument("--continuations-file", type=str, default="simpleqa_continuations.jsonl", help="File to save/load continuations")
    parser.add_argument("--continuation-responses-file", type=str, default="simpleqa_continuations_responses.jsonl", help="File to save/load continuation responses")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    
    if args.mode == "generate":
        # Generate initial answers
        print("Generating initial answers...")
        generate_initial_answers(
            num_examples=args.num_examples,
            output_jsonl=args.responses_file
        )
    
    elif args.mode == "consensus":
        # Update responses with consensus
        print("Updating responses with consensus...")
        update_responses_with_consensus(
            responses_jsonl="simpleqa_responses.jsonl",
            output_jsonl="simpleqa_consensus_responses.jsonl"
        )
    
    elif args.mode == "continuations":
        # Generate thinking continuations
        print("Generating thinking continuations...")
        generate_thinking_continuations(
            input_jsonl=args.responses_file, 
            output_jsonl=args.continuations_file, 
            num_continuations=args.num_continuations
        )
    
    elif args.mode == "process":
        # Process thinking continuations
        print("Processing thinking continuations...")
        process_continuations(
            continuations_jsonl=args.continuations_file,
            output_jsonl=args.continuation_responses_file
        )
    
    elif args.mode == "evaluate":
        # Evaluate answers
        print("Evaluating answers...")
        evaluate_simpleqa_answers(
            eval_jsonl="simpleqa_consensus_responses.jsonl"
        )
if __name__ == "__main__":
    main()

