"""
create a Whoosh BM25 index from the chunked text data in the CSV
for fast keyword-based search functionality
"""

import os
import pandas as pd
import json
import ast
from datetime import datetime
from tqdm import tqdm
import logging

# Whoosh imports
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Term, And, Or
from whoosh import scoring

def setup_logging():
    """Setup logging for the indexing process"""
    log_file = "logs/whoosh_indexing.log"
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def find_latest_csv():
    """Find the latest CSV file with embeddings"""
    csv_dir = "input/csv"

    if not os.path.exists(csv_dir):
        print(f"CSV directory not found: {csv_dir}")
        return None

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return None

    # Prefer files with 'embeddings' in the name
    embedding_csvs = [f for f in csv_files if 'embedding' in f.lower()]

    if embedding_csvs:
        embedding_csvs.sort()
        latest_csv = embedding_csvs[-1]
    else:
        csv_files.sort()
        latest_csv = csv_files[-1]

    csv_path = os.path.join(csv_dir, latest_csv)
    print(f"Using CSV file: {csv_path}")

    return csv_path

def create_whoosh_schema():
    """Create Whoosh schema for book chunks"""

    # Use stemming analyzer for better search recall
    analyzer = StemmingAnalyzer()

    schema = Schema(
        # Unique identifier for each chunk
        chunk_id=ID(stored=True, unique=True),

        # Main searchable text content (most important for BM25)
        text=TEXT(analyzer=analyzer, stored=True, phrase=False),

        # Book metadata (searchable and stored)
        book_title=TEXT(analyzer=analyzer, stored=True),
        book_author=TEXT(analyzer=analyzer, stored=True),
        book_filename=TEXT(analyzer=analyzer, stored=True),
        book_importance=KEYWORD(stored=True),  # Exact match for filtering
        book_topic=KEYWORD(stored=True),       # Exact match for filtering

        # Additional metadata
        public_url=STORED(),
        embed_filename=STORED(),

        # Page information
        page_nums=STORED(),

        # For result ranking/filtering
        chunk_length=STORED(),

        # Google Books metadata (searchable)
        g_books_description=TEXT(analyzer=analyzer, stored=True),
        g_books_categories=TEXT(analyzer=analyzer, stored=True),

        # Scholar data (searchable)
        scholar_snippet=TEXT(analyzer=analyzer, stored=True),

        # Search data (searchable)
        search_snippet=TEXT(analyzer=analyzer, stored=True)
    )

    return schema

def parse_embed_id_dict(embed_dict_str):
    """Parse embed_id_dict from CSV string to Python dict"""
    if not embed_dict_str or embed_dict_str == '' or pd.isna(embed_dict_str):
        return {}

    try:
        # Try JSON first
        return json.loads(embed_dict_str)
    except json.JSONDecodeError:
        try:
            # Try ast.literal_eval
            return ast.literal_eval(embed_dict_str)
        except (ValueError, SyntaxError):
            try:
                # Last resort: eval
                return eval(embed_dict_str)
            except:
                print(f"Could not parse embed_id_dict: {embed_dict_str[:100]}...")
                return {}

def extract_chunks_from_csv(csv_path, logger):
    """Extract all chunks from CSV with book metadata"""

    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} books")

    # Check for embed_id_dict column
    if 'embed_id_dict' not in df.columns:
        logger.error("embed_id_dict column not found in CSV")
        return []

    # Count books with embeddings
    books_with_embeddings = df[
        df['embed_id_dict'].notna() &
        (df['embed_id_dict'] != '') &
        (df['embed_id_dict'] != '{}')
    ]

    logger.info(f"Books with embeddings: {len(books_with_embeddings)}/{len(df)}")
    print(f"Books with embeddings: {len(books_with_embeddings)}/{len(df)}")

    # Extract all chunks
    all_chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing books"):
        # Parse embed_id_dict
        embed_dict = parse_embed_id_dict(row.get('embed_id_dict', '{}'))

        if not embed_dict:
            continue  # Skip books without embeddings

        # Extract book metadata
        book_metadata = {
            'book_title': str(row.get('Title', '')),
            'book_author': str(row.get('Author', '')),
            'book_filename': str(row.get('filename', '')),
            'book_importance': str(row.get('Importance', '')),
            'book_topic': str(row.get('Topic', '')),
            'public_url': str(row.get('public_url', '')),
            'embed_filename': str(row.get('embed_filename', '')),
            'g_books_description': str(row.get('description', '')),
            'g_books_categories': str(row.get('categories', '')),
            'scholar_snippet': str(row.get('scholar_snippet', '')),
            'search_snippet': str(row.get('search_snippet', ''))
        }

        # Process each chunk
        for chunk_id, chunk_text in embed_dict.items():
            if not chunk_text or chunk_text.strip() == '':
                continue

            # Extract page numbers from chunk_id
            # Format: filename_book.json__page_[0,1,2]__chunk_3
            page_nums = ""
            try:
                if '__page_' in chunk_id:
                    page_part = chunk_id.split('__page_')[1].split('__chunk_')[0]
                    page_nums = page_part.strip('[]')
            except:
                page_nums = "unknown"

            # Create chunk document
            chunk_doc = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'page_nums': page_nums,
                'chunk_length': len(chunk_text.split()),
                **book_metadata
            }

            all_chunks.append(chunk_doc)

    logger.info(f"Extracted {len(all_chunks)} chunks for indexing")
    print(f"Extracted {len(all_chunks)} chunks for indexing")

    return all_chunks

def create_whoosh_index(chunks, output_dir, logger):
    """Create Whoosh index from chunks"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create schema
    schema = create_whoosh_schema()

    # Create index directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    index_name = f"books_whoosh_{date_str}"
    index_path = os.path.join(output_dir, index_name)

    # Create the specific index directory
    os.makedirs(index_path, exist_ok=True)

    print(f"Creating Whoosh index at: {index_path}")
    logger.info(f"Creating Whoosh index: {index_path}")

    # Create the index
    ix = index.create_in(index_path, schema)

    # Add documents to index
    writer = ix.writer()

    print("Adding chunks to Whoosh index...")
    for chunk in tqdm(chunks, desc="Indexing chunks"):
        try:
            writer.add_document(**chunk)
        except Exception as e:
            logger.error(f"Error indexing chunk {chunk.get('chunk_id', 'unknown')}: {e}")

    # Commit the changes
    print("Committing index...")
    writer.commit()

    logger.info(f"Successfully created Whoosh index with {len(chunks)} documents")
    print(f"Successfully created Whoosh index with {len(chunks)} documents")

    return index_path

def test_whoosh_index(index_path, logger):
    """Test the created Whoosh index with sample queries"""

    print("\nTesting Whoosh index...")

    try:
        # Open the index
        ix = index.open_dir(index_path)

        # Test queries
        test_queries = [
            "contract law",
            "constitutional rights",
            "legal principles",
            "court decision"
        ]

        with ix.searcher(weighting=scoring.BM25F()) as searcher:
            # Create multi-field parser
            parser = MultifieldParser(
                ["text", "book_title", "book_author", "g_books_description"],
                ix.schema
            )

            for query_str in test_queries:
                print(f"\nTesting query: '{query_str}'")

                try:
                    query = parser.parse(query_str)
                    results = searcher.search(query, limit=3)

                    print(f"Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result['book_title']} (Score: {result.score:.3f})")
                        print(f"     Chunk: {result['chunk_id']}")
                        print(f"     Text preview: {result['text'][:100]}...")

                except Exception as e:
                    print(f"Error with query '{query_str}': {e}")

        logger.info("Whoosh index testing completed successfully")
        print(" Whoosh index testing completed!")

    except Exception as e:
        logger.error(f"Error testing Whoosh index: {e}")
        print(f"Error testing index: {e}")

def save_index_metadata(index_path, chunks_count, logger):
    """Save metadata about the created index"""

    metadata = {
        "index_path": index_path,
        "creation_date": datetime.now().isoformat(),
        "total_chunks": chunks_count,
        "schema_fields": [
            "chunk_id", "text", "book_title", "book_author",
            "book_filename", "book_importance", "book_topic",
            "public_url", "page_nums", "chunk_length"
        ],
        "analyzer": "StemmingAnalyzer",
        "scoring": "BM25F"
    }

    metadata_file = os.path.join(os.path.dirname(index_path), "index_metadata.json")

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Index metadata saved: {metadata_file}")
    print(f"Index metadata saved: {metadata_file}")

def main():
    """Main Whoosh indexing function"""

    print("Starting Whoosh BM25 Index Creation")
    print("=" * 50)

    # Setup logging
    logger = setup_logging()
    logger.info("Starting Whoosh indexing process")

    # Find CSV file
    csv_file = find_latest_csv()
    if not csv_file:
        logger.error("No CSV file found")
        return

    # Extract chunks from CSV
    chunks = extract_chunks_from_csv(csv_file, logger)
    if not chunks:
        logger.error("No chunks found for indexing")
        print("No chunks found. Make sure your CSV has books with embed_id_dict populated.")
        return

    # Create output directory
    output_dir = "output/whoosh"

    # Create Whoosh index
    index_path = create_whoosh_index(chunks, output_dir, logger)

    # Test the index
    test_whoosh_index(index_path, logger)

    # Save metadata
    save_index_metadata(index_path, len(chunks), logger)

    # Final summary
    print("\n" + "=" * 50)
    print("Whoosh BM25 Index Creation Complete!")
    print("=" * 50)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Index location: {index_path}")
    print(f"Metadata: {os.path.join(output_dir, 'index_metadata.json')}")

    logger.info("Whoosh indexing process completed successfully")

if __name__ == "__main__":
    main()