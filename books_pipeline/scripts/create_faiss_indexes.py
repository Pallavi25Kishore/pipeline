
"""
Read embeddings pickle, process metadata, and create FAISS indexes
"""

import os
import pickle
import numpy as np
import collections
import faiss
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import defaultdict
import sys
from datetime import datetime

def load_embeddings_pickle():
    """
    Load embeddings from the intermediate pickle file
    """
    pickles_path = "intermediate/embeddings/bodies"
    pickle_file = os.path.join(pickles_path, "books_embeddings.pkl")

    if not os.path.exists(pickle_file):
        print(f"Embeddings pickle not found: {pickle_file}")
        print("Run embedding creation first: python scripts/03_embed_books.py")
        return None

    print(f"Loading embeddings from: {pickle_file}")

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    embeddings = data['embeddings'].numpy()  # Convert from torch tensor
    chunk_ids = data['ids']

    print(f"Loaded {len(chunk_ids)} embeddings with shape {embeddings.shape}")

    # Convert to list format expected by processing functions
    all_data = []
    for embedding, chunk_id in zip(embeddings, chunk_ids):
        all_data.append({
            "embedding": embedding,
            "chunk_id": chunk_id
        })

    return all_data

def add_metadata(item):
    """
    Add metadata to each embedding item following docs_to_faiss pattern
    """
    chunk_id = item['chunk_id']

    # Parse chunk_id: "filename_book.json__page_[0,1,2]__chunk_3"
    try:
        # Extract components
        if chunk_id.startswith('filename_'):
            # Remove 'filename_' prefix
            remaining = chunk_id[9:]  # Remove 'filename_'

            # Split on '__page_'
            parts = remaining.split('__page_')
            if len(parts) == 2:
                file_part = parts[0]
                page_chunk_part = parts[1]

                # Split page and chunk parts
                page_chunk_split = page_chunk_part.split('__chunk_')
                if len(page_chunk_split) == 2:
                    page_part = page_chunk_split[0]
                    chunk_num = page_chunk_split[1]

                    # Extract page numbers from [0,1,2] format
                    if page_part.startswith('[') and page_part.endswith(']'):
                        pages_str = page_part[1:-1]  # Remove brackets
                        if ',' in pages_str:
                            pages = pages_str  # Keep as comma-separated string
                        else:
                            pages = pages_str  # Single page
                    else:
                        pages = page_part

                    file_id = file_part
                else:
                    # Fallback parsing
                    pages = "0"
                    file_id = remaining
            else:
                # Fallback parsing
                pages = "0"
                file_id = remaining
        else:
            # Fallback for unexpected format
            pages = "0"
            file_id = chunk_id

    except Exception as e:
        print(f"Error parsing chunk_id {chunk_id}: {e}")
        pages = "0"
        file_id = "unknown"

    # Update item with metadata
    item.update({
        "page_nums": pages,
        "file_id": file_id,
        "chunk_id": chunk_id
    })

    return item

def load_chunk_text_content():
    """
    Load chunk text content from JSON files or CSV
    """
    print("Loading chunk text content...")

    # Try to load from cleaned JSON files first
    json_dir = "intermediate/cleaned/jsons"
    chunk_text_map = {}

    if os.path.exists(json_dir):
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

        for json_file in tqdm(json_files, desc="Loading text content"):
            json_path = os.path.join(json_dir, json_file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    book_data = json.load(f)

                # This contains page_num -> {pure_ocr_doc, lazy_clean_doc, legacy_clean_doc}
                # use legacy_clean_doc as the content
                for page_num, page_content in book_data.items():
                    text_content = page_content.get('legacy_clean_doc', '')

                    page_key = f"filename_{json_file}__page_[{page_num}]"
                    chunk_text_map[page_key] = text_content

            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    # Also try to load from CSV embed_id_dict
    csv_dir = "input/csv"
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csv_files:
            csv_files.sort()
            latest_csv = csv_files[-1]
            csv_path = os.path.join(csv_dir, latest_csv)

            try:
                import pandas as pd
                df = pd.read_csv(csv_path)

                if 'embed_id_dict' in df.columns:
                    for _, row in df.iterrows():
                        embed_dict = row.get('embed_id_dict', '{}')
                        if embed_dict and embed_dict != '{}':
                            try:
                                if isinstance(embed_dict, str):
                                    embed_dict = eval(embed_dict)  # Convert string to dict
                                chunk_text_map.update(embed_dict)
                            except:
                                continue

            except Exception as e:
                print(f"Error loading CSV content: {e}")

    print(f"Loaded text content for {len(chunk_text_map)} chunks")
    return chunk_text_map

def process_embeddings_with_content(all_data, chunk_text_map):
    """
    Add text content to embedding items
    """
    print("Adding text content to embeddings...")

    missing_content = 0

    for item in tqdm(all_data, desc="Adding content"):
        chunk_id = item['chunk_id']

        # Try exact match first
        if chunk_id in chunk_text_map:
            item['highlight'] = chunk_text_map[chunk_id]
        else:
            # Try partial matches for different formats
            content_found = False
            for key, content in chunk_text_map.items():
                if chunk_id in key or key in chunk_id:
                    item['highlight'] = content
                    content_found = True
                    break

            if not content_found:
                item['highlight'] = ""
                missing_content += 1

    if missing_content > 0:
        print(f"{missing_content} chunks missing content")

    return all_data

def create_final_arrays(all_data):
    """
    Create final embedding arrays and metadata following docs_to_faiss pattern
    """
    print("Creating final arrays...")

    num_items = len(all_data)
    dim = 768  # Embedding dimension

    # Preallocate numpy array for embeddings
    embeddings = np.empty((num_items, dim), dtype=np.float32)

    # Initialize metadata ordered dict
    metadata = collections.OrderedDict()

    # Counters for missing data
    count_missing_chunks = 0
    count_missing_pages = 0
    count_missing_files = 0
    count_missing_highlights = 0

    for i in tqdm(range(num_items), desc="Processing items"):
        item = all_data[i]

        # Store embedding
        embeddings[i] = item['embedding']

        # Store metadata
        metadata[i] = {
            'chunk_id': item.get('chunk_id', ''),
            'page_nums': item.get('page_nums', ''),
            'file_id': item.get('file_id', ''),
            'highlight': item.get('highlight', ''),
        }

        # Update counters
        if not item.get('highlight'):
            count_missing_highlights += 1
        if not item.get('file_id'):
            count_missing_files += 1
        if not item.get('page_nums'):
            count_missing_pages += 1
        if not item.get('chunk_id'):
            count_missing_chunks += 1

    print(f"Final metadata summary:")
    print(f"   Total items: {num_items}")
    print(f"   Missing highlights: {count_missing_highlights}")
    print(f"   Missing files: {count_missing_files}")
    print(f"   Missing pages: {count_missing_pages}")
    print(f"   Missing chunks: {count_missing_chunks}")

    return embeddings, metadata

def save_final_outputs(embeddings, metadata):
    """
    Save final embeddings, metadata, and FAISS indexes
    """
    print("Saving final outputs...")

    # Create output directories
    output_dirs = [
        './output/embeddings',
        './output/ids',
        './output/indexes'
    ]

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    # Create date string for filenames
    date = datetime.now().strftime("%Y-%m-%d")

    # Save embeddings
    embeddings_file = f'./output/embeddings/books_embeddings_{date}.npy'
    np.save(embeddings_file, embeddings)
    print(f"Embeddings saved: {embeddings_file}")

    # Save metadata
    metadata_file = f'./output/ids/books_metadata_{date}.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved: {metadata_file}")

    # Clear metadata from memory
    del metadata

    # Create FAISS indexes
    print("Building FAISS indexes...")

    dim = embeddings.shape[1]

    # Flat Index for inner product (IP) similarity
    print("   Building Flat IP index...")
    flat = faiss.IndexFlatIP(dim)
    flat.add(embeddings)

    flat_index_file = f'./output/indexes/books_flat_ip_{date}.index'
    faiss.write_index(flat, flat_index_file)
    print(f"Flat IP index saved: {flat_index_file}")

    del flat

    #Create HNSW index for faster search
    print("   Building HNSW index...")
    M = 64
    ef_construction = 128
    ef_search = 64

    hnsw = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = ef_construction
    hnsw.hnsw.efSearch = ef_search
    hnsw.add(embeddings)

    hnsw_index_file = f'./output/indexes/books_hnsw_{date}.index'
    faiss.write_index(hnsw, hnsw_index_file)
    print(f"HNSW index saved: {hnsw_index_file}")

    print(f"\nIndex creation complete!")
    print(f"Output files:")
    print(f"   Embeddings: {embeddings_file}")
    print(f"   Metadata: {metadata_file}")
    print(f"   FAISS Index: {flat_index_file}")

def main():
    """
    Main index creation function
    """
    print("Starting FAISS Index Creation for Books")

    # Step 1: Load embeddings pickle
    all_data = load_embeddings_pickle()
    if all_data is None:
        return

    # Step 2: Add metadata to each item
    print("Adding metadata...")
    all_data = [add_metadata(item) for item in tqdm(all_data, desc="Adding metadata")]

    # Step 3: Load chunk text content
    chunk_text_map = load_chunk_text_content()

    # Step 4: Add text content to embeddings
    all_data = process_embeddings_with_content(all_data, chunk_text_map)

    # Step 5: Create final arrays
    embeddings, metadata = create_final_arrays(all_data)

    # Step 6: Save outputs and create indexes
    save_final_outputs(embeddings, metadata)

    print(f"\n➡️  Next step: python scripts/05_load_to_db.py")

if __name__ == "__main__":
    main()