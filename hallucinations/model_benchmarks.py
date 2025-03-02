#!/usr/bin/env python3
"""
SimpleQA evaluation script for comparing Claude and OpenAI models.
"""

import os
import json
import argparse
import random
import re
import asyncio
import subprocess
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import anthropic
import openai
import dotenv

# Import from simple-evals
from simple_evals.simpleqa_eval import (
    GRADER_TEMPLATE, 
    CHOICE_LETTER_TO_STRING, 
    SimpleQAEval
)
from simple_evals.sampler.chat_completion_sampler import ChatCompletionSampler

dotenv.load_dotenv()

async def get_questions(num_examples=100, start_example=None, end_example=None):
    """Load SimpleQA questions using the simple-evals library approach."""
    # Load data from the same source as SimpleQAEval
    df = pd.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    )
    
    # Convert to list of dictionaries
    questions = [row.to_dict() for _, row in df.iterrows()]
    
    # Select a specific range if provided
    if start_example is not None and end_example is not None:
        # Handle 1-based indexing from command line args
        start_idx = max(0, start_example - 1)
        end_idx = min(len(questions), end_example)
        
        if start_idx < end_idx:
            questions = questions[start_idx:end_idx]
        else:
            print(f"Invalid range: {start_example}-{end_example}, using all questions")
    # Otherwise select a random subset if requested
    elif 0 < num_examples < len(questions):
        random.seed(42)  # For reproducibility
        questions = random.sample(questions, num_examples)
    
    # Format for consistency
    formatted_questions = []
    for i, q in enumerate(questions):
        formatted_questions.append({
            "question_id": str(i+1),
            "question": q.get("problem", ""),
            "gold_answer": q.get("answer", "")
        })
    
    return formatted_questions

async def generate_claude_answers(questions, model="claude-3-opus-20240229", 
                                 output_file="simpleqa_claude_responses.jsonl"):
    """Generate answers from Claude."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create output file or clear it if it exists
    with open(output_file, "w") as f:
        pass
    
    responses = []
    for i, q in tqdm(enumerate(questions), total=len(questions), desc="Getting Claude answers"):
        question_id = q.get("question_id", str(i+1))
        question_text = q.get("question", "")
        gold_answer = q.get("gold_answer", "")
        
        try:
            # Generate response from Claude
            message = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": question_text}
                ]
            )
            
            response_text = message.content[0].text
            
            # Store results
            result = {
                "question_id": question_id,
                "question": question_text,
                "gold_answer": gold_answer,
                "predicted_answer": response_text,
                "model": model,
                "timestamp": datetime.now().isoformat()
            }
            responses.append(result)
            
            # Write as we go to avoid losing data
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                
            print(f"Question {question_id}: {'Answer: ' + response_text[:100]}...")
                
        except Exception as e:
            print(f"Error with Claude on question {question_id}: {e}")
    
    return output_file

async def generate_openai_answers(questions, model="gpt-4o", 
                                output_file="simpleqa_openai_responses.jsonl"):
    """Generate answers from OpenAI."""
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Create output file or clear it if it exists
    with open(output_file, "w") as f:
        pass
    
    responses = []
    batch_size = 5  # Process in small batches
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
        
        tasks = []
        for q in batch:
            question_id = q.get("question_id")
            question_text = q.get("question", "")
            gold_answer = q.get("gold_answer", "")
            
            print(f"Queueing question {question_id}: {question_text[:100]}...")
            
            task = asyncio.create_task(client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": question_text}
                ],
                temperature=0.5,
                max_tokens=1024
            ))
            tasks.append((task, question_id, question_text, gold_answer))
        
        for task, question_id, question_text, gold_answer in tqdm(tasks, desc="Getting OpenAI answers"):
            try:
                response = await task
                response_text = response.choices[0].message.content
                
                # Store results
                result = {
                    "question_id": question_id,
                    "question": question_text,
                    "gold_answer": gold_answer,
                    "predicted_answer": response_text,
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                }
                responses.append(result)
                
                # Write as we go to avoid losing data
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                
                print(f"Question {question_id}: {'Answer: ' + response_text[:100]}...")
                    
            except Exception as e:
                print(f"Error with OpenAI on question {question_id}: {e}")
    
    return output_file

async def evaluate_answers(eval_jsonl):
    """Evaluate answers using GPT-4o as the grader."""
    if not os.path.exists(eval_jsonl):
        print(f"Evaluation file {eval_jsonl} does not exist.")
        return None
    
    # Initialize grader model
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
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
    
    # Process and grade entries in batches
    batch_size = 10  # Adjust based on rate limits
    print(f"Evaluating {len(eval_entries)} responses from {eval_jsonl}...")
    
    for i in range(0, len(eval_entries), batch_size):
        batch = eval_entries[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(eval_entries) + batch_size - 1)//batch_size}")
        
        # Create grading tasks
        grading_tasks = []
        for entry in batch:
            question_id = entry.get("question_id", "unknown")
            question_text = entry.get("question", "")
            gold_answer = entry.get("gold_answer", "")
            predicted_answer = entry.get("predicted_answer", "")
            
            # Skip if no gold answer
            if not gold_answer:
                print(f"Skipping question {question_id}: No gold answer")
                continue
            
            grader_prompt = GRADER_TEMPLATE.format(
                question=question_text,
                target=gold_answer,
                predicted_answer=predicted_answer,
            )
            
            task = asyncio.create_task(client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": grader_prompt}]
            ))
            grading_tasks.append((task, question_id, question_text))
        
        # Process grading results
        for task, question_id, question_text in tqdm(grading_tasks, desc="Grading answers"):
            try:
                response = await task
                response_text = response.choices[0].message.content
                match = re.search(r"(A|B|C)", response_text)
                grade_letter = match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
                grade_string = CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")
                
                # Update results
                results[grade_string] += 1
                
                print(f"Question {question_id}: Graded as {grade_string}")
                
            except Exception as e:
                print(f"Error grading question {question_id}: {e}")
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
    print("\n========== RESULTS FOR", eval_jsonl, "==========")
    print(f"Examples evaluated: {total}")
    print(f"Correct: {results['CORRECT']} ({is_correct:.2%})")
    print(f"Incorrect: {results['INCORRECT']} ({is_incorrect:.2%})")
    print(f"Not Attempted: {results['NOT_ATTEMPTED']} ({is_not_attempted:.2%})")
    print(f"Accuracy Given Attempted: {accuracy_given_attempted:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=============================")
    
    return {
        "file": eval_jsonl,
        "model": eval_entries[0].get("model", "unknown") if eval_entries else "unknown",
        "total": total,
        "correct": results["CORRECT"],
        "incorrect": results["INCORRECT"],
        "not_attempted": results["NOT_ATTEMPTED"],
        "accuracy_given_attempted": accuracy_given_attempted,
        "f1": f1
    }

async def compare_results(results):
    """Compare results between models."""
    if len(results) < 2:
        print("Not enough results to compare.")
        return
    
    # Create a comparison table
    df = pd.DataFrame(results)
    
    # Set model as the index
    df = df.set_index('model')
    
    print("\n========== MODEL COMPARISON ==========")
    print(df[['correct', 'incorrect', 'not_attempted', 'accuracy_given_attempted', 'f1']])
    print("=======================================")
    
    # Determine better model
    best_model = df['f1'].idxmax()
    print(f"\nBest performing model: {best_model} (F1: {df.loc[best_model, 'f1']:.4f})")

async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate SimpleQA questions with Claude and OpenAI")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples to process")
    parser.add_argument("--start-example", type=int, help="Start index for example range (1-based)")
    parser.add_argument("--end-example", type=int, help="End index for example range (1-based, inclusive)")
    parser.add_argument("--claude-model", type=str, default="claude-3-5-sonnet-20240620", help="Claude model name")
    parser.add_argument("--openai-model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument("--questions-file", type=str, default="simpleqa.jsonl", help="JSONL file with questions")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load questions
    questions = await get_questions(
        num_examples=args.num_examples,
        start_example=args.start_example,
        end_example=args.end_example
    )
    if not questions:
        print("No questions loaded. Exiting.")
        return
    
    range_info = ""
    if args.start_example and args.end_example:
        range_info = f" (range: {args.start_example}-{args.end_example})"
    
    print(f"Loaded {len(questions)} questions for evaluation{range_info}.")
    
    # Generate answers with both models
    claude_file = await generate_claude_answers(
        questions, 
        model=args.claude_model,
        output_file=f"simpleqa_{args.claude_model.replace('-', '_')}_responses_test.jsonl"
    )
    
    openai_file = await generate_openai_answers(
        questions, 
        model=args.openai_model,
        output_file=f"simpleqa_{args.openai_model.replace('-', '_')}_responses_test.jsonl"
    )
    
    # Evaluate both sets of answers
    results = []
    claude_results = await evaluate_answers(claude_file)
    if claude_results:
        results.append(claude_results)
    
    openai_results = await evaluate_answers(openai_file)
    if openai_results:
        results.append(openai_results)
    
    # Compare results
    await compare_results(results)

def main():
    """Entry point for the script."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()