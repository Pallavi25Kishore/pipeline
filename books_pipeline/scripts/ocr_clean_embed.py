
"""
ocr_clean_embed.py

Automatically runs:
1. OCR processing
2. Text cleaning
3. Embeddings creation
"""

import subprocess
import sys
import time
import os
from datetime import datetime
import signal

interrupted = False
current_process = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global interrupted, current_process
    interrupted = True

    print("\n\n" + "="*80)
    print("PIPELINE INTERRUPTED BY USER (Ctrl+C)")
    print("="*80)

    if current_process and current_process.poll() is None:
        print("Terminating current subprocess...")
        current_process.terminate()
        try:
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing subprocess...")
            current_process.kill()

    print("\nCleaning up and exiting...")
    sys.exit(1)

def run_command(command, step_name):
    """Run a command and handle its output"""
    global current_process, interrupted

    if interrupted:
        return False

    print(f"\n{'='*80}")
    print(f"Starting Step: {step_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    try:

        current_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Print output in real-time
        for line in iter(current_process.stdout.readline, ''):
            if interrupted:
                break
            print(line, end='')

        # Wait for completion
        return_code = current_process.wait()
        current_process = None  # Clear when done

        if interrupted:
            return False
        elif return_code == 0:
            print(f"\n{step_name} completed successfully!")
            return True
        else:
            print(f"\n{step_name} failed with return code: {return_code}")
            return False

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
        return False
    except Exception as e:
        print(f"\nError running {step_name}: {e}")
        return False

def check_requirements():
    """Check if required directories and files exist"""
    checks = [
        ("input/pdfs", "PDF input directory"),
        ("input/csv", "CSV input directory"),
        ("scripts/ocr.py", "OCR script"),
        ("scripts/clean_txt.py", "Text cleaning script"),
        ("scripts/embed_books.py", "Embedding script")
    ]

    all_good = True
    for path, description in checks:
        if os.path.exists(path):
            print(f"{description} found: {path}")
        else:
            print(f"{description} NOT found: {path}")
            all_good = False

    return all_good

def main():
    """Main pipeline runner"""
    print("\n" + "="*80)
    print("BOOKS PROCESSING PIPELINE")
    print("="*80)
    signal.signal(signal.SIGINT, signal_handler)

    # Check requirements
    print("\nChecking requirements...")
    if not check_requirements():
        print("\nMissing required files/directories. Please check your setup.")
        sys.exit(1)

    # Record start time
    pipeline_start = time.time()

    # Step 1: OCR Processing
    if not run_command([sys.executable, "scripts/ocr.py"], "OCR Processing"):
        print("\nPipeline stopped due to OCR failure")
        sys.exit(1)

    # Step 2: Text Cleaning
    if not run_command([sys.executable, "scripts/clean_txt.py"], "Text Cleaning"):
        print("\nPipeline stopped due to text cleaning failure")
        sys.exit(1)

    # Step 3: Embeddings Creation
    if not run_command([sys.executable, "scripts/embed_books.py"], "Embeddings Creation"):
        print("\nPipeline stopped due to embeddings creation failure")
        sys.exit(1)

    # Calculate total time
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    # Success summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nTotal pipeline time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nResults:")
    print(f"  - OCR text files: intermediate/ocr/txts/")
    print(f"  - Cleaned JSON files: intermediate/cleaned/jsons/")
    print(f"  - Embeddings: intermediate/embeddings/bodies/books_embeddings.pkl")
    print(f"  - Updated CSV: input/csv/books_with_embeddings.csv")

    # Check final results
    if os.path.exists("input/csv/books_with_embeddings.csv"):
        import pandas as pd
        df = pd.read_csv("input/csv/books_with_embeddings.csv")
        books_with_embeddings = sum(1 for d in df['embed_id_dict']
                                  if pd.notna(d) and d != '' and d != '{}')
        print(f"\nBooks with embeddings: {books_with_embeddings}/{len(df)}")

if __name__ == "__main__":
    main()