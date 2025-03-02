#!/usr/bin/env python3
"""
Refactor simpleqa_consensus_responses.jsonl to match the format of simpleqa_responses.jsonl.
"""

import json
import argparse
import os
from datetime import datetime

def refactor_consensus_responses(input_jsonl, output_jsonl):
    """
    Refactor consensus responses JSONL to the format of initial responses JSONL.

    Args:
        input_jsonl: Path to the input consensus responses JSONL file.
        output_jsonl: Path to the output refactored responses JSONL file.
    """

    refactored_entries = []
    with open(input_jsonl, 'r') as f_in:
        for line in f_in:
            try:
                entry = json.loads(line.strip())
                question_id = entry.get('question_id')
                question = entry.get('question')
                gold_answer = entry.get('gold_answer')

                # Extract content from existing response format
                if 'response' in entry and isinstance(entry['response'], dict) and 'choices' in entry['response']:
                    content = entry['response']['choices'][0]['message']['content']
                    content = content.split("</think>", 1)[1].strip()
                else:
                    print(f"Skipping entry with missing content: {entry}")
                    continue

                if question_id and question and content:
                    refactored_entry = {
                        "question_id": question_id,
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": content
                    }
                    refactored_entries.append(refactored_entry)

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    with open(output_jsonl, 'w') as f_out:
        for refactored_entry in refactored_entries:
            f_out.write(json.dumps(refactored_entry) + '\n')

    print(f"Refactored {len(refactored_entries)} entries and saved to {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactor consensus responses JSONL format.")
    parser.add_argument("--input-jsonl", type=str, default="simpleqa_responses.jsonl",
                        help="Path to the input simpleqa_consensus_responses.jsonl file.")
    parser.add_argument("--output-jsonl", type=str, default="simpleqa_responses_eval.jsonl",
                        help="Path to the output refactored simpleqa_responses.jsonl file.")
    args = parser.parse_args()

    refactor_consensus_responses(args.input_jsonl, args.output_jsonl) 