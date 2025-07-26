import pandas as pd
import os

def add_new_books():
    """Add new books from Excel to the embeddings CSV"""

    # Paths
    new_books_excel = 'input/csv/new_books.xlsx'
    embeddings_csv = 'input/csv/books_with_embeddings.csv'

    # Check if new books file exists
    if not os.path.exists(new_books_excel):
        print(f"No new books file found at {new_books_excel}")
        return

    # Read new books from Excel
    print(f"Reading new books from {new_books_excel}")
    new_books = pd.read_excel(new_books_excel)

    # Read just the columns from existing CSV (1 row to get structure)
    existing_columns = pd.read_csv(embeddings_csv, nrows=1).columns.tolist()

    # Add missing columns to new books
    for col in existing_columns:
        if col not in new_books.columns:
            if col in ['embed_id_dict', 'embed_filename']:
                new_books[col] = ''  # Empty for embedding columns
            else:
                new_books[col] = None  # None for other missing columns

    # Ensure column order matches
    new_books = new_books[existing_columns]

    # Append to CSV
    new_books.to_csv(embeddings_csv,
                    mode='a',        # Append mode
                    header=False,    # No headers
                    index=False)

    print(f"Added {len(new_books)} new books to {embeddings_csv}")

    # # Optionally, archive the processed Excel file
    # archive_name = new_books_excel.replace('.xlsx', f'_processed_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    # os.rename(new_books_excel, archive_name)
    # print(f"Archived processed file to {archive_name}")

if __name__ == "__main__":
    add_new_books()