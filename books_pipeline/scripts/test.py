import pandas as pd
df = pd.read_csv('input/csv/books_with_embeddings.csv')

print("Column names:", df.columns.tolist())
print("\nFirst row Title:", df['Title'].iloc[0])
print("Type:", type(df['Title'].iloc[0]))
print("Is it NaN?:", pd.isna(df['Title'].iloc[0]))

print("\nFirst few values:")
for col in ['Title', 'Author', 'filename']:
    print(f"{col}: {df[col].iloc[0]}")