
import nltk
import os
import shutil

print("Fixing NLTK data...")

# Remove corrupted NLTK data
nltk_data_dir = os.path.expanduser('~/nltk_data')
if os.path.exists(nltk_data_dir):
    print(f"Removing old NLTK data from {nltk_data_dir}")
    shutil.rmtree(nltk_data_dir)
    print("Old data removed")

# Re-download fresh data
print("\nDownloading fresh NLTK data...")
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    print("\n✓ NLTK data successfully downloaded!")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Test if it works
print("\nTesting NLTK...")
try:
    from nltk.corpus import wordnet
    from nltk.corpus import stopwords

    # Test wordnet
    test = wordnet.synsets('test')
    print(f"✓ WordNet working - found {len(test)} synsets for 'test'")

    # Test stopwords
    stops = stopwords.words('english')
    print(f"✓ Stopwords working - loaded {len(stops)} English stopwords")

    print("\nNLTK is fixed and ready to use!")

except Exception as e:
    print(f"❌ NLTK test failed: {e}")