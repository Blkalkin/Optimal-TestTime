import json

def calculate_average_text_length(file_path):
    total_length = 0
    count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    # Parse the JSON object
                    data = json.loads(line)
                    
                    # Get the text field from the first choice in the response
                    text = data['response']['choices'][0]['text']
                    
                    # Add the length to the total
                    total_length += len(text)
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing line: {e}")
    
    # Calculate the average length
    average_length = total_length / count if count > 0 else 0
    
    return average_length, count

# Run the function with the file path
file_path = 'hallucination_experiment_data/250/simpleqa_continuations_responses_test_250.jsonl'
average_length, count = calculate_average_text_length(file_path)

print(f"Average length of the 'text' field: {average_length:.2f} characters")
print(f"Number of JSON objects processed: {count}")
