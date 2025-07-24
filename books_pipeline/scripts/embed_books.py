
"""
embed_books.py - Book Chunking and Embedding

Reads cleaned JSON files, chunks them page-aware, creates embeddings,
and generates final CSV with embed_id_dict populated.

"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch
import pickle
import time
from multiprocessing import Pool
import sys
import gc

def split_text(df, text_column, doc_id_column, para_id_column, max_len=350, min_len=175, overlap=50):
    """
    Split text into chunks with overlap, maintaining page awareness
    """
    new_texts = []
    new_doc_ids = []
    new_para_ids = []

    buffer_text = []
    buffer_doc_id = None
    buffer_para_ids = []

    for i in reversed(df.index):
        row = df.loc[i]
        doc_id = row[doc_id_column]
        para_id = row[para_id_column]
        text = str(row[text_column]).split()

        prev_doc_id = df.loc[i-1, doc_id_column] if i-1 >= 0 else np.nan

        # If doc_id changes, clear the buffer
        if doc_id != buffer_doc_id and buffer_text:
            new_texts.insert(0, ' '.join(buffer_text))
            new_doc_ids.insert(0, buffer_doc_id)
            new_para_ids.insert(0, buffer_para_ids)
            buffer_text = []
            buffer_para_ids = []

        buffer_para_ids.insert(0, para_id)
        buffer_doc_id = doc_id

        if buffer_text:
            if len(buffer_text) + len(text) <= max_len:
                text = text + buffer_text
                buffer_text = []

        while len(text) > max_len:
            new_texts.insert(0, ' '.join(text[-max_len:]))
            text = text[:-max_len+overlap]
            new_doc_ids.insert(0, doc_id)
            new_para_ids.insert(0, buffer_para_ids)
            buffer_para_ids = [para_id]

        if len(text) >= min_len or doc_id != prev_doc_id:
            new_texts.insert(0, ' '.join(text))
            new_doc_ids.insert(0, doc_id)
            new_para_ids.insert(0, buffer_para_ids)
            buffer_text = []
            buffer_para_ids = []
        else:
            buffer_text = text

    # Handle remaining buffer
    if buffer_text:
        new_texts.insert(0, ' '.join(buffer_text))
        new_doc_ids.insert(0, buffer_doc_id)
        new_para_ids.insert(0, buffer_para_ids)

    # Create new DataFrame
    new_df = pd.DataFrame({
        text_column: new_texts,
        doc_id_column: new_doc_ids,
        para_id_column: new_para_ids
    })

    # Add length column
    new_df['length'] = new_df[text_column].apply(lambda x: len(x.split()))

    # Ensure overlap and clean up
    new_df = ensure_overlap(new_df, text_column, doc_id_column, para_id_column, max_len, overlap)

    # Update length column
    new_df['length'] = new_df[text_column].apply(lambda x: len(x.split()))
    new_df[para_id_column] = new_df[para_id_column].apply(lambda x: sorted(x))

    return new_df

def ensure_overlap(df, text_column, doc_id_column, para_id_column, max_len, overlap):
    """
    Ensure proper overlap between chunks and remove too-small chunks
    """
    # Processing in forward direction
    for i in tqdm(range(len(df) - 1), desc="Adding overlap"):
        if df.loc[i+1, para_id_column][0] != 0:
            df.loc[i+1, text_column] = ' '.join(df.loc[i, text_column].split()[-overlap:] + df.loc[i+1, text_column].split())

    df['length'] = df[text_column].apply(lambda x: len(x.split()))

    # Processing in reverse direction - remove small chunks
    indices_to_drop = []
    for i in tqdm(reversed(df.index), desc="Removing small chunks"):
        if (df.loc[i, 'length'] < overlap and i > 0 and
            df.loc[i, doc_id_column] == df.loc[i-1, doc_id_column] and
            df.loc[i, para_id_column][0] != 0):
            df.loc[i-1, para_id_column].extend(df.loc[i, para_id_column])
            indices_to_drop.append(i)
    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['length'] = df[text_column].apply(lambda x: len(x.split()))

    # Handle first rows of each document
    for i in tqdm(df.index, desc="Cleaning first pages"):
        if (df.loc[i, para_id_column][0] == 0 and df.loc[i, 'length'] < overlap and
            i < len(df)-1 and df.loc[i+1, para_id_column][0] != 0):
            df.loc[i+1, para_id_column].extend(df.loc[i, para_id_column])
            df.drop(i, inplace=True)

    return df

def process_cleaned_jsons(batch_size=3):
    """
    Load cleaned JSON files and create DataFrame for chunking
    """
    input_dir = "intermediate/cleaned/jsons"

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        print("   Run text cleaning first: python scripts/02_clean_texts.py")
        return None

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    completed_json_files = check_existing_embeddings()
    if completed_json_files:
        json_files = [f for f in json_files if f not in completed_json_files]
        print(f"Will process {len(json_files)} new books")

    if not json_files:
        print("No new JSON files to process - all books already have embeddings!")
        return None

    print(f"Loading {len(json_files)} cleaned JSON files in batches of {batch_size}...")

    # Process in batches
    all_book_embed_dicts = {}

    for batch_start in range(0, len(json_files), batch_size):
        batch_end = min(batch_start + batch_size, len(json_files))
        batch_files = json_files[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start//batch_size + 1}: Books {batch_start+1}-{batch_end} of {len(json_files)}")
        print(f"{'='*60}")

        # Create DataFrame from this batch
        dfs = []

        for json_file in tqdm(batch_files, desc=f"Loading batch {batch_start//batch_size + 1}"):
            filepath = os.path.join(input_dir, json_file)

            with open(filepath, 'r', encoding='utf-8') as f:
                book_data = json.load(f)

            # Convert to DataFrame format
            pages = []
            page_nums = []
            filenames = []

            for page_num, page_content in book_data.items():
                text = page_content.get('legacy_clean_doc', '')
                pages.append(text)
                page_nums.append(int(page_num))
                filenames.append(json_file)

            book_df = pd.DataFrame({
                'text': pages,
                'page_num': page_nums,
                'filename': filenames
            })

            dfs.append(book_df)

        # Combine this batch
        batch_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(batch_df)} pages from {len(batch_files)} books in this batch")

        # Process this batch through the pipeline
        process_batch(batch_df, all_book_embed_dicts)

        # Clear memory
        del dfs, batch_df
        gc.collect()

        print(f"Batch {batch_start//batch_size + 1} complete. Memory freed.")

    return all_book_embed_dicts

def check_existing_embeddings():
    """
    Check which books already have embeddings in books_with_embeddings.csv
    """
    csv_dir = "input/csv"
    embeddings_csv = os.path.join(csv_dir, 'books_with_embeddings.csv')

    # If embeddings CSV doesn't exist, no books have embeddings yet
    if not os.path.exists(embeddings_csv):
        print("No existing embeddings found (books_with_embeddings.csv doesn't exist)")
        return set()

    try:
        df = pd.read_csv(embeddings_csv)

        # Check which books have non-empty embed_id_dict
        if 'embed_id_dict' in df.columns:
            completed_books = df[
                df['embed_id_dict'].notna() &
                (df['embed_id_dict'] != '') &
                (df['embed_id_dict'] != '{}')
            ]['filename'].tolist()

            # Convert to expected JSON filenames
            completed_json_files = set()
            for filename in completed_books:
                possible_names = [
                    filename.replace('.pdf', '.json'),
                    filename.replace('.PDF', '.json'),
                    filename.lower().replace('.pdf', '.json')
                ]
                completed_json_files.update(possible_names)

            print(f"Found {len(completed_books)} books with existing embeddings in books_with_embeddings.csv")
            return completed_json_files
    except Exception as e:
        print(f"Could not read embeddings CSV: {e}")

    return set()

def setup_embedding_model():
    """
    Set up the embedding model and device
    """
    print("Setting up embedding model...")

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # Setup device
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    model = model.to(device)

    return tokenizer, model, device

def cls_pooling(model_output):
    """CLS Pooling - Take output from first token"""
    return model_output.last_hidden_state[:,0]

def encode_text(text, tokenizer, model, device):
    """
    Encode single text to embedding
    """
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Alert if text is longer than 512 tokens
    if len(encoded_input['input_ids'][0]) > 512:
        print('⚠️  Text is longer than 512 tokens.')

    # Move tensors to device
    for key in encoded_input:
        encoded_input[key] = encoded_input[key].to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = cls_pooling(model_output)

    # Move embeddings to cpu
    embeddings = embeddings.cpu()

    return embeddings

def create_embeddings(chunks_df):
    """
    Create embeddings for all chunks
    """
    print("Creating embeddings...")

    # Setup model
    tokenizer, model, device = setup_embedding_model()

    # Create embeddings
    embeddings_list = []
    chunk_ids = []

    for i, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Creating embeddings"):
        # Create chunk ID
        if isinstance(row['page_num'], list):
            page_nums_str = ', '.join(map(str, row['page_num']))
        else:
            page_nums_str = str(row['page_num'])

        chunk_id = f"filename_{row['filename']}__page_[{page_nums_str}]__chunk_{row['chunk_id']}"

        # Get embedding
        embedding = encode_text(row['text'], tokenizer, model, device)

        embeddings_list.append(embedding)
        chunk_ids.append(chunk_id)

        # Clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings_list, chunk_ids

def create_embed_id_dict(chunks_df):
    """
    Create embed_id_dict mapping chunk_id to text content for each book
    """
    print("Creating embed_id_dict for each book...")

    book_embed_dicts = {}

    # Group by filename
    grouped = chunks_df.groupby('filename')

    for filename, group in tqdm(grouped, desc="Processing books"):
        embed_dict = {}

        for _, row in group.iterrows():
            # Create chunk ID
            if isinstance(row['page_num'], list):
                page_nums_str = ', '.join(map(str, row['page_num']))
            else:
                page_nums_str = str(row['page_num'])
            chunk_id = f"filename_{filename}__page_[{page_nums_str}]__chunk_{row['chunk_id']}"

            # Store text content
            embed_dict[chunk_id] = row['text']

        book_embed_dicts[filename] = embed_dict

    return book_embed_dicts

def update_csv_with_embeddings(book_embed_dicts):
    """
    Update the CSV with embed_id_dict and embed_filename
    Always reads from books.csv and writes to books_with_embeddings.csv
    """
    print("Updating CSV with embedding data...")

    csv_dir = "input/csv"

    # Always read from the original books.csv
    original_csv = os.path.join(csv_dir, 'books.csv')
    embeddings_csv = os.path.join(csv_dir, 'books_with_embeddings.csv')

    if not os.path.exists(original_csv):
        print(f"Error: {original_csv} not found!")
        return None

    # Check if embeddings CSV already exists
    if os.path.exists(embeddings_csv):
        print(f"Found existing {embeddings_csv}, will update it")
        df = pd.read_csv(embeddings_csv)
    else:
        print(f"Creating new embeddings CSV from {original_csv}")
        df = pd.read_csv(original_csv)
        # Initialize embedding columns if they don't exist
        if 'embed_id_dict' not in df.columns:
            df['embed_id_dict'] = ''
        if 'embed_filename' not in df.columns:
            df['embed_filename'] = ''

    # Update only rows that don't have embeddings yet
    updated_count = 0

    for idx, row in df.iterrows():
        # Skip if already has embeddings
        if (pd.notna(row.get('embed_id_dict')) and
            row.get('embed_id_dict') != '' and
            row.get('embed_id_dict') != '{}'):
            continue

        filename = str(row['filename'])

        # Try different filename formats
        possible_names = [
            filename.replace('.pdf', '.json'),
            filename.replace('.PDF', '.json'),
            filename.lower().replace('.pdf', '.json')
        ]

        for name in possible_names:
            if name in book_embed_dicts:
                df.at[idx, 'embed_id_dict'] = book_embed_dicts[name]
                df.at[idx, 'embed_filename'] = name
                updated_count += 1
                break

    # Always save to books_with_embeddings.csv
    df.to_csv(embeddings_csv, index=False)

    print(f"Updated {updated_count} new books with embeddings")
    print(f"Saved to: {embeddings_csv}")

    # Print summary
    books_with_embeddings = sum(1 for d in df['embed_id_dict']
                               if pd.notna(d) and d != '' and d != '{}')
    print(f"Total: {books_with_embeddings}/{len(df)} books now have embeddings")

    return embeddings_csv

def save_embeddings_batch(embeddings_list, chunk_ids):
    """
    Save embeddings in append mode for batch processing
    """
    print("Saving batch embeddings...")

    output_dir = "intermediate/embeddings/bodies"
    os.makedirs(output_dir, exist_ok=True)

    # Convert embeddings to tensor
    embeddings_tensor = torch.cat(embeddings_list, dim=0)

    # Load existing embeddings if they exist
    output_file = os.path.join(output_dir, 'books_embeddings.pkl')

    if os.path.exists(output_file):
        print("Loading existing embeddings to append...")
        with open(output_file, 'rb') as f:
            existing_data = pickle.load(f)

        # Append new data
        combined_embeddings = torch.cat([existing_data['embeddings'], embeddings_tensor], dim=0)
        combined_ids = existing_data['ids'] + chunk_ids
    else:
        combined_embeddings = embeddings_tensor
        combined_ids = chunk_ids

    # Save combined data
    embedding_data = {
        'embeddings': combined_embeddings,
        'ids': combined_ids
    }

    with open(output_file, 'wb') as f:
        pickle.dump(embedding_data, f)

    print(f"Batch saved. Total embeddings now: {combined_embeddings.shape}")

def process_batch(batch_df, all_book_embed_dicts):
    """
    Process a batch of books through chunking and embedding
    """
    # Step 1: Create chunks
    print("Creating chunks for this batch...")
    chunks_df = split_text(batch_df, text_column='text',
                          doc_id_column='filename',
                          para_id_column='page_num')

    # Step 2: Add chunk IDs
    print("Adding chunk IDs...")
    chunks_df['chunk_id'] = 0
    chunks_df.reset_index(drop=True, inplace=True)

    # Assign sequential chunk IDs per book
    for i in tqdm(range(len(chunks_df)-1), desc="Assigning chunk IDs"):
        if (chunks_df.loc[i, 'filename'] == chunks_df.loc[i+1, 'filename'] and
            chunks_df.loc[i, 'page_num'] == chunks_df.loc[i+1, 'page_num']):
            chunks_df.loc[i+1, 'chunk_id'] = chunks_df.loc[i, 'chunk_id'] + 1

    print(f"Created {len(chunks_df)} chunks in this batch")
    print(f"Average chunk length: {chunks_df['length'].mean():.1f} words")

    # Step 3: Create embeddings
    embeddings_list, chunk_ids = create_embeddings(chunks_df)

    # Step 4: Save embeddings (append mode)
    save_embeddings_batch(embeddings_list, chunk_ids)

    # Step 5: Create embed_id_dict for this batch
    batch_embed_dicts = create_embed_id_dict(chunks_df)

    # Add to overall dictionary
    all_book_embed_dicts.update(batch_embed_dicts)

    # Clear memory
    del chunks_df, embeddings_list, chunk_ids
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

def main():
    """
    Main embedding pipeline function
    """
    print("Starting Book Chunking and Embedding Pipeline")

    # Add garbage collection
    import gc

    # Process in batches of 5 books
    all_book_embed_dicts = process_cleaned_jsons(batch_size=3)

    if all_book_embed_dicts is None:
        return

    # Update CSV with all embed_id_dicts
    updated_csv = update_csv_with_embeddings(all_book_embed_dicts)

    print(f"\nEmbedding Pipeline Complete!")
    print(f"Updated CSV: {updated_csv}")
    print(f"Embeddings saved for FAISS and Whoosh index creation")

if __name__ == "__main__":
    main()