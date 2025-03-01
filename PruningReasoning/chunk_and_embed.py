#!/usr/bin/env python3
import json
import os
import glob
import argparse
import tiktoken
import numpy as np
from openai import OpenAI
from tqdm import tqdm

def load_responses(file_path):
    """Load responses from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problem_id = f"{data.get('year', 'unknown')}_{data.get('number', 'unknown')}"
    prompt = data.get('prompt', '')
    responses = data.get('responses', [])
    
    return problem_id, prompt, responses

def chunk_text(text, encoding_name="cl100k_base", chunk_size=300):
    """Split text into chunks of approximately chunk_size tokens."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def batch_embed_chunks(client, chunks, model="text-embedding-3-large", batch_size=100):
    """Embed chunks in batches using OpenAI's embedding model."""
    all_embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float"
        )
        
        # Extract embeddings from response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def process_file(file_path, client, output_dir, chunk_size=300, embedding_model="text-embedding-3-large", specific_chunks=None):
    """Process a single file: chunk responses and create embeddings.
    
    Args:
        file_path: Path to the JSON file with responses
        client: OpenAI client
        output_dir: Directory to save results
        chunk_size: Number of tokens per chunk
        embedding_model: OpenAI embedding model to use
        specific_chunks: List of specific chunk indexes to process (e.g., [0, 1, 2, 3])
                         If None, all chunks will be processed
    """
    try:
        problem_id, prompt, responses = load_responses(file_path)
        
        file_results = {
            "problem_id": problem_id,
            "prompt": prompt,
            "responses": []
        }
        
        all_chunks = []
        chunk_metadata = []
        
        # Process each response
        for response_idx, response in enumerate(responses):
            response_text = response.get('text', '')
            
            # Chunk the response
            response_chunks = chunk_text(response_text, chunk_size=chunk_size)
            
            # Store metadata for each chunk, filtering by specific_chunks if provided
            for chunk_idx, chunk in enumerate(response_chunks):
                if specific_chunks is None or chunk_idx in specific_chunks:
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "response_idx": response_idx,
                        "chunk_idx": chunk_idx,
                        "text": chunk
                    })
        
        # Batch embed all chunks
        print(f"Embedding {len(all_chunks)} chunks for {problem_id}...")
        embeddings = batch_embed_chunks(client, all_chunks, model=embedding_model)
        
        # Organize results by response
        response_data = {}
        for metadata, embedding in zip(chunk_metadata, embeddings):
            response_idx = metadata["response_idx"]
            chunk_idx = metadata["chunk_idx"]
            
            if response_idx not in response_data:
                response_data[response_idx] = {
                    "chunks": [],
                    "chunk_indices": [],
                    "embeddings": []
                }
            
            response_data[response_idx]["chunks"].append(metadata["text"])
            response_data[response_idx]["chunk_indices"].append(chunk_idx)
            response_data[response_idx]["embeddings"].append(embedding)
        
        # Format the final results
        for response_idx, data in response_data.items():
            file_results["responses"].append({
                "response_idx": response_idx,
                "chunks": data["chunks"],
                "chunk_indices": data["chunk_indices"],
                "embeddings": data["embeddings"]
            })
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(file_results, f)
        
        print(f"Successfully processed {file_path}")
        return file_results
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Chunk responses and create embeddings')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--input_pattern', type=str, default='responses/*.json', help='Glob pattern for input files')
    parser.add_argument('--output_dir', type=str, default='chunked_embeddings', help='Directory to save chunked embeddings')
    parser.add_argument('--chunk_size', type=int, default=300, help='Number of tokens per chunk')
    parser.add_argument('--embedding_model', type=str, default='text-embedding-3-large', help='OpenAI embedding model to use')
    parser.add_argument('--specific_chunks', type=str, help='Comma-separated list of specific chunk indexes to process (e.g., "0,1,2,3")')
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Provide it via --api_key argument or OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    # Parse specific chunks if provided
    specific_chunks = None
    if args.specific_chunks:
        try:
            specific_chunks = [int(chunk) for chunk in args.specific_chunks.split(',')]
            print(f"Processing only specific chunks: {specific_chunks}")
        except ValueError:
            print(f"Error parsing specific_chunks '{args.specific_chunks}'. Processing all chunks.")
    
    # Process all matching files
    files = glob.glob(args.input_pattern)
    print(f"Found {len(files)} files to process")
    
    for file_path in tqdm(files, desc="Processing files"):
        process_file(
            file_path, 
            client, 
            args.output_dir, 
            chunk_size=args.chunk_size,
            embedding_model=args.embedding_model,
            specific_chunks=specific_chunks
        )
    
    print(f"Done! Chunked embeddings saved to {args.output_dir}")

if __name__ == "__main__":
    main() 