import json
import sys
import os
import tempfile

def clean_responses_inplace(input_file):
    """
    Filter JSONL file in-place to keep only entries containing "</think>" in their text.
    
    Args:
        input_file (str): Path to JSONL file to clean
    """
    kept_count = 0
    total_count = 0
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp_file:
        temp_filename = temp_file.name
        
        # Read from original file and write filtered content to temp file
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                total_count += 1
                try:
                    entry = json.loads(line.strip())
                    text = entry['response']['choices'][0]['text']
                    
                    if '</think>' in text:
                        temp_file.write(line)
                        kept_count += 1
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error processing line: {e}")
                    continue
    
    # Replace the original file with the temporary file
    os.replace(temp_filename, input_file)
    
    print(f"Processed {total_count} entries")
    print(f"Kept {kept_count} entries ({kept_count/total_count:.2%})")
    print(f"Removed {total_count - kept_count} entries")

if __name__ == "__main__":
    input_file = "simpleqa_continuations_responses_500.jsonl"
    
    # Allow command-line argument to override default file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    clean_responses_inplace(input_file)
