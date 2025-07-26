import pickle
import pandas as pd
import os

# Check pickle file
with open('intermediate/embeddings/bodies/books_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total embeddings: {data['embeddings'].shape}")
print(f"Total chunk IDs: {len(data['ids'])}")

import pickle
import pandas as pd
import json

# Load embeddings
with open('intermediate/embeddings/bodies/books_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Load CSV
df = pd.read_csv('input/csv/books_with_embeddings.csv')
books_with_embeddings = df[df['embed_id_dict'].notna() & (df['embed_id_dict'] != '') & (df['embed_id_dict'] != '{}')]

# Get last book
last_book = books_with_embeddings.iloc[-1]
last_book_name = last_book['embed_filename']  # The JSON filename

# Find chunks from this book in the pickle file
last_book_chunks = [chunk_id for chunk_id in data['ids'] if chunk_id.startswith(f"filename_{last_book_name}__")]

print(f"Last book: {last_book_name}")
print(f"Number of chunks: {len(last_book_chunks)}")
print(f"First 3 chunks:")
for chunk in last_book_chunks[:3]:
    print(f"  - {chunk}")
print(f"Last 3 chunks:")
for chunk in last_book_chunks[-3:]:
    print(f"  - {chunk}")