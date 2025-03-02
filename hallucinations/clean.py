import json
import re
import argparse

def extract_answer(content):
    """Extract the actual answer, removing the thinking part if present."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    return content

def parse_jsonl_for_evaluation(input_file, output_file):
    """
    Parse the input JSONL file and convert it to the format expected by evaluate_simpleqa_answers.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to write the formatted output JSONL file
    """
    print(f"Reading from {input_file}")
    processed_entries = 0
    
    with open(output_file, 'w') as out_f:
        with open(input_file, 'r') as in_f:
            for line in in_f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    entry = json.loads(line)
                    
                    # Extract the predicted answer from the response
                    predicted_answer = ""
                    if "response" in entry and "choices" in entry["response"] and len(entry["response"]["choices"]) > 0:
                        content = entry["response"]["choices"][0]["message"]["content"]
                        predicted_answer = extract_answer(content)
                    
                    # Create the formatted entry
                    formatted_entry = {
                        "question_id": entry.get("question_id", "unknown"),
                        "question": entry.get("question", ""),
                        "gold_answer": entry.get("gold_answer", ""),
                        "predicted_answer": predicted_answer
                    }
                    
                    # Write to output file
                    out_f.write(json.dumps(formatted_entry) + "\n")
                    processed_entries += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    print(f"Successfully processed {processed_entries} entries")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse SimpleQA responses for evaluation")
    parser.add_argument("--input", type=str, default="simpleqa_responses_test.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output", type=str, default="simpleqa_eval.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()
    
    parse_jsonl_for_evaluation(args.input, args.output)