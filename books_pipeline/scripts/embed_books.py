
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
    for i in tqdm(reversed(df.index), desc="Removing small chunks"):
        if (df.loc[i, 'length'] < overlap and i > 0 and
            df.loc[i, doc_id_column] == df.loc[i-1, doc_id_column] and
            df.loc[i, para_id_column][0] != 0):
            df.loc[i-1, para_id_column].extend(df.loc[i, para_id_column])
            df.drop(i, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df['length'] = df[text_column].apply(lambda x: len(x.split()))

    # Handle first rows of each document
    for i in tqdm(df.index, desc="Cleaning first pages"):
        if (df.loc[i, para_id_column][0] == 0 and df.loc[i, 'length'] < overlap and
            i < len(df)-1 and df.loc[i+1, para_id_column][0] != 0):
            df.loc[i+1, para_id_column].extend(df.loc[i, para_id_column])
            df.drop(i, inplace=True)

    return df

def process_cleaned_jsons():
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
        print(f"Found {len(completed_json_files)} books with existing embeddings")
        json_files = [f for f in json_files if f not in completed_json_files]
        print(f"Will process {len(json_files)} new books")

    if not json_files:
        print("No new JSON files to process - all books already have embeddings!")
        return None

    print(f"Loading {len(json_files)} cleaned JSON files...")

    # Create DataFrame from JSON files
    dfs = []

    for json_file in tqdm(json_files, desc="Loading JSON files"):
        filepath = os.path.join(input_dir, json_file)

        with open(filepath, 'r', encoding='utf-8') as f:
            book_data = json.load(f)

        # Convert to DataFrame format
        pages = []
        page_nums = []
        filenames = []

        for page_num, page_content in book_data.items():
            # Use legacy_clean_doc as the main text
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

    # Combine all books
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} pages from {len(json_files)} books")

    return combined_df

def check_existing_embeddings():
    """
    Check which books already have embeddings in the CSV
    """
    csv_dir = "input/csv"
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        return set()

    # Find the most recent CSV (with or without embeddings)
    csv_files.sort()
    embedding_csvs = [f for f in csv_files if 'embedding' in f.lower()]
    if embedding_csvs:
        csv_file = embedding_csvs[-1]
    else:
        csv_file = csv_files[-1]

    csv_path = os.path.join(csv_dir, csv_file)

    try:
        df = pd.read_csv(csv_path)
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
                    f"{filename}.json",
                    filename.replace('.pdf', '.json'),
                    filename.replace('&', '_').replace("'", '_') + '.json'
                ]
                completed_json_files.update(possible_names)

            return completed_json_files
    except Exception as e:
        print(f"Could not read CSV: {e}")

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
        page_nums_str = str(row['page_num']).replace(' ', '')
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
            page_nums_str = str(row['page_num']).replace(' ', '').replace(',', ', ')
            chunk_id = f"filename_{filename}__page_[{page_nums_str}]__chunk_{row['chunk_id']}"

            # Store text content
            embed_dict[chunk_id] = row['text']

        book_embed_dicts[filename] = embed_dict

    return book_embed_dicts

def update_csv_with_embeddings(book_embed_dicts):
    """
    Update the original CSV with embed_id_dict and embed_filename
    """
    print("Updating CSV with embedding data...")

    # Find CSV file
    csv_dir = "input/csv"
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    # Use the first CSV file (or you can specify which one)
    csv_file = csv_files[0]
    csv_path = os.path.join(csv_dir, csv_file)

    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_path)

    # Update CSV with embedding data
    def update_embed_data(row):
        filename = str(row['filename'])

        # Try different filename formats
        possible_names = [
            f"{filename}.json",
            filename.replace('.pdf', '.json'),
            filename.replace('.PDF', '.json'),
            filename.lower().replace('.pdf', '.json'),  # Case insensitive
            filename.replace('&', '_').replace("'", '_') + '.json' if not filename.endswith('.json') else filename
        ]

        for name in possible_names:
            if name in book_embed_dicts:
                return book_embed_dicts[name], name

        return {}, ''

    # Only update rows that don't have embeddings yet
    def update_row_if_needed(row):
        # Skip if already has embeddings
        if (pd.notna(row.get('embed_id_dict')) and
            row.get('embed_id_dict') != '' and
            row.get('embed_id_dict') != '{}'):
            return row['embed_id_dict'], row.get('embed_filename', '')

        # Update if no embeddings
        return update_embed_data(row)

    tqdm.pandas(desc="Updating new CSV rows")
    embed_data = df.progress_apply(update_row_if_needed, axis=1)

    df['embed_id_dict'] = [data[0] for data in embed_data]
    df['embed_filename'] = [data[1] for data in embed_data]

    # Save updated CSV
    output_csv = os.path.join(csv_dir, csv_file.replace('.csv', '_with_embeddings.csv'))
    df.to_csv(output_csv, index=False)

    print(f"Updated CSV saved: {output_csv}")

    # Print summary
    books_with_embeddings = sum(1 for d in df['embed_id_dict'] if d != {})
    print(f"Summary: {books_with_embeddings}/{len(df)} books have embeddings")

    return output_csv

def save_embeddings(embeddings_list, chunk_ids):
    """
    Save embeddings and IDs for FAISS index creation
    """
    print("Saving embeddings...")

    # Create output directory
    output_dir = "intermediate/embeddings/bodies"
    os.makedirs(output_dir, exist_ok=True)

    # Convert embeddings to tensor
    embeddings_tensor = torch.cat(embeddings_list, dim=0)

    # Save as pickle
    embedding_data = {
        'embeddings': embeddings_tensor,
        'ids': chunk_ids
    }

    output_file = os.path.join(output_dir, 'books_embeddings.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(embedding_data, f)

    print(f"Embeddings saved: {output_file}")
    print(f"Shape: {embeddings_tensor.shape}")
    print(f"Total chunks: {len(chunk_ids)}")

def main():
    """
    Main embedding pipeline function
    """
    print("Starting Book Chunking and Embedding Pipeline")

    # Step 1: Load cleaned JSON files
    df = process_cleaned_jsons()
    if df is None:
        return

    # Step 2: Create chunks with page awareness
    print("Creating chunks with page awareness...")
    chunks_df = split_text(df, text_column='text', doc_id_column='filename', para_id_column='page_num')

    # Step 3: Add chunk IDs
    print("Adding chunk IDs...")
    chunks_df['chunk_id'] = 0
    chunks_df.reset_index(drop=True, inplace=True)

    # Assign sequential chunk IDs per book
    for i in tqdm(range(len(chunks_df)-1), desc="Assigning chunk IDs"):
        if (chunks_df.loc[i, 'filename'] == chunks_df.loc[i+1, 'filename'] and
            chunks_df.loc[i, 'page_num'] == chunks_df.loc[i+1, 'page_num']):
            chunks_df.loc[i+1, 'chunk_id'] = chunks_df.loc[i, 'chunk_id'] + 1

    print(f"Created {len(chunks_df)} chunks")
    print(f"Average chunk length: {chunks_df['length'].mean():.1f} words")

    # Step 4: Create embeddings
    embeddings_list, chunk_ids = create_embeddings(chunks_df)

    # Step 5: Save embeddings
    save_embeddings(embeddings_list, chunk_ids)

    # Step 6: Create embed_id_dict for CSV
    book_embed_dicts = create_embed_id_dict(chunks_df)

    # Step 7: Update CSV
    updated_csv = update_csv_with_embeddings(book_embed_dicts)

    print(f"\nEmbedding Pipeline Complete!")
    print(f"Updated CSV: {updated_csv}")
    print(f"Embeddings saved for FAISS and Whoosh index creation")

if __name__ == "__main__":
    main()