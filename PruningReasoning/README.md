# Answer Extraction Tool

This tool extracts answers from JSON files containing math problem solutions by:
1. Reading JSON files with a specific structure
2. Extracting the last 500 characters from each "text" field in the responses
3. Sending these characters to OpenAI's GPT-4o model to extract the final answer
4. Saving the results in a structured JSON format

## Directory Structure

- `responses/`: Contains the original JSON files with math problem solutions
- `extracted_answers/`: Contains the extracted answers from each response
- `chunked_embeddings/`: Contains chunked text and embeddings for semantic search

## Requirements

- Python 3.6+
- OpenAI Python package
- tiktoken (for chunking)
- tqdm (for progress bars)
- scikit-learn (for similarity search and clustering)
- numpy (for vector operations)
- matplotlib (for plotting)

Install the required packages:

```bash
pip install openai tiktoken tqdm scikit-learn numpy matplotlib
```

## Usage

### Extracting Answers

```bash
python extract_answers.py --api_key YOUR_OPENAI_API_KEY --input_pattern "responses/*.json" --chars 500
```

#### Arguments

- `--api_key`: Your OpenAI API key (optional if OPENAI_API_KEY environment variable is set)
- `--input_pattern`: Glob pattern for input files (default: "responses/*.json")
- `--chars`: Number of characters to extract from the end of each response (default: 500)

#### Environment Variables

You can also set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Comparing with Ground Truth

After extracting answers, you can compare them with ground truth answers from a CSV file:

```bash
python compare_answers.py --csv AIME_2023_4.csv --extracted extracted_answers --output comparison_report.md
```

#### Arguments

- `--csv`: Path to the CSV file with ground truth answers (default: "AIME_2023_4.csv")
- `--extracted`: Directory containing extracted answer files (default: "extracted_answers")
- `--output`: Output file for the comparison report (default: "comparison_report.md")

### Chunking and Embedding Responses

To chunk responses into 300-token segments and create embeddings:

```bash
python chunk_and_embed.py --api_key YOUR_OPENAI_API_KEY --input_pattern "responses/*.json" --output_dir chunked_embeddings --chunk_size 300
```

#### Arguments

- `--api_key`: Your OpenAI API key (optional if OPENAI_API_KEY environment variable is set)
- `--input_pattern`: Glob pattern for input files (default: "responses/*.json")
- `--output_dir`: Directory to save chunked embeddings (default: "chunked_embeddings")
- `--chunk_size`: Number of tokens per chunk (default: 300)
- `--embedding_model`: OpenAI embedding model to use (default: "text-embedding-3-large")


### Clustering Embeddings

To perform k-means clustering on embeddings and evaluate if correct answers are present in the clustered responses:

```bash
python cluster_embeddings.py --csv AIME_2023_4.csv --extracted extracted_answers --embeddings chunked_embeddings --cluster_sizes "10,15,20,25,30,35,40,45" --midway_only --max_problems 3
```

#### Arguments

- `--csv`: Path to the CSV file with ground truth answers (default: "AIME_2023_4.csv")
- `--extracted`: Directory containing extracted answer files (default: "extracted_answers")
- `--embeddings`: Directory containing chunked embeddings (default: "chunked_embeddings")
- `--output`: Output file for the evaluation report (default: "clustering_report.md")
- `--plot`: Output file for the accuracy plot (default: "clustering_accuracy.png")
- `--cluster_sizes`: Comma-separated list of cluster sizes to evaluate (default: "10,15,20,25,30,35,40,45")
- `--midway_only`: Only process chunks up to the midway point of each problem (reduces processing time)
- `--max_problems`: Maximum number of problems to process (default: all problems)

### Calculating pass@k Metrics

To estimate theoretical pass@k performance based on the number of correct samples per problem:

```bash
python calculate_pass_at_k.py --csv AIME_2023_4.csv --extracted extracted_answers --n 100 --k_values "1,5,10,25,50,100" --output pass_at_k.png
```

#### Arguments

- `--csv`: Path to the CSV file with ground truth answers (default: "AIME_2023_4.csv")
- `--extracted`: Directory containing extracted answer files (default: "extracted_answers")
- `--n`: Total number of samples per problem (default: 100)
- `--k_values`: Comma-separated list of k values to calculate pass@k for (default: "1,5,10,25,50,100")
- `--output`: Output file for the pass@k plot (default: "pass_at_k.png")

This script implements the unbiased estimation formula from Chen et al. for calculating pass@k:

pass@k = (1 / num_problems) * sum_{i=1}^{num_problems} (1 - (n-c_i choose k) / (n choose k))

Where:
- n is the total number of samples per problem
- c_i is the number of correct samples for problem i
- k is the number of samples to consider

The script generates a plot showing the estimated pass@k performance for different values of k, which helps predict how performance would scale with more samples.

## Output

### Extraction Output

The extraction script creates an `extracted_answers` directory containing JSON files with the same names as the input files. Each output file contains:

- `problem_id`: Constructed from the "year" and "number" fields in the input
- `prompt`: The original problem statement
- `results`: An array of objects, each containing:
  - `response_index`: The index of the response in the original file
  - `extracted_answer`: The answer extracted by GPT-4o

### Comparison Output

The comparison script generates a Markdown report with:

- Overall statistics on accuracy
- Problem-by-problem breakdown of correct and incorrect answers
- Detailed comparison between ground truth and extracted answers

### Chunking and Embedding Output

The chunking and embedding script creates a `chunked_embeddings` directory containing JSON files with:

- `problem_id`: Constructed from the "year" and "number" fields in the input
- `prompt`: The original problem statement
- `responses`: An array of objects, each containing:
  - `response_idx`: The index of the response in the original file
  - `chunks`: An array of text chunks (approximately 300 tokens each)
  - `embeddings`: An array of vector embeddings corresponding to each chunk


### Clustering Output

The clustering script generates:

1. A simplified Markdown report (`clustering_simplified_report.md`) containing:
   - For each problem:
     - The ground truth answer
     - The best cluster size based on silhouette score
     - A table showing whether the correct answer is present (✓) or absent (✗) for each combination of chunk index and cluster size
     - A summary row showing whether any chunk has the correct answer for each cluster size
   - An aggregate statistics section that shows:
     - For each chunk index and cluster size combination, how many problems had at least one correct answer (displayed as both a ratio and percentage)
     - Maximum values highlighted for easy identification:
       - **Bold** values indicate maximum correctness for each chunk index (row)
       - *Italic* values indicate maximum correctness for each cluster size (column)
       - Values that are maximum for both row and column are shown in **bold**
     - A summary row showing, for each cluster size, how many problems had a correct answer in any chunk, with the best overall cluster sizes in **bold**

2. A plot (`clustering_accuracy.png`) showing:
   - The success rate (percentage of problems where at least one chunk contains the correct answer) for different numbers of clusters
   - The silhouette scores for different numbers of clusters, helping to identify the optimal cluster size

This simplified report format makes it easy to see at a glance which combinations of chunk index and cluster size preserve the correct answer, which is essential for determining if pruning via clustering is effective. The aggregate statistics help identify the most reliable chunk indices and cluster sizes across all problems, with highlighting to quickly spot the best combinations.

## Example

For an input file `responses/deepseek_2023_1.json`, the extraction output might look like:

```json
{
  "problem_id": "2023_1",
  "prompt": "The numbers of apples growing on each of six apple trees...",
  "results": [
    {
      "response_index": 0,
      "extracted_answer": "220"
    },
    {
      "response_index": 1,
      "extracted_answer": "220"
    }
  ]
}
```

## Workflow

A typical workflow using these scripts would be:

1. Extract answers from JSON files using `extract_answers.py`
2. Compare extracted answers with ground truth using `compare_answers.py`
3. Chunk and embed responses using `chunk_and_embed.py`
5. Cluster embeddings and evaluate effectiveness using `cluster_embeddings.py` 