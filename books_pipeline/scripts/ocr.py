import os
import cv2
import numpy as np
import pytesseract
import tempfile
from multiprocessing import Pool, Value, Lock, Manager
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import time
from functools import partial
import io
import subprocess
import pandas as pd
import sys
import json
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in GB"""
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

def get_available_memory():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / 1024 / 1024 / 1024

def denoise(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	denoised = cv2.medianBlur(gray, 1) # median filter
	return denoised

def sharpen(image):
	kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	sharpened = cv2.filter2D(image, -1, kernel)
	return sharpened

def compress_pdf(source_file, dest_file):
	subprocess.call([
            'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
            '-dPDFSETTINGS=/ebook', '-dNOPAUSE', '-dBATCH', '-dQUIET',
            '-sOutputFile={}'.format(dest_file), source_file
        ])

def process_single_page(args):
    """Process a single page - for parallel processing"""
    pdf_path, page_num, out_dir, filename = args

    try:
        # Convert only the specific page
        pages = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        if not pages:
            return None

        page = pages[0]

        # Convert to numpy array and process
        page_array = np.array(page.convert('RGB'))
        page_array = denoise(page_array)
        page_array = sharpen(page_array)

        # Extract text
        txt_content = pytesseract.image_to_string(page_array, lang='eng', config='--psm 6')

        # Save text
        page_out_path = os.path.join(out_dir, "txts", filename, f'page_{page_num}.txt')
        with open(page_out_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)

        # Clean up
        del pages
        del page_array
        gc.collect()

        return page_num

    except Exception as e:
        print(f"Error processing page {page_num} of {filename}: {e}")
        return None

def process_pdf_file(args):
    try:
        out, file = args
        filename = os.path.splitext(os.path.basename(file))[0]

        print(f"\nProcessing: {filename}")
        print(f"Current memory usage: {get_memory_usage():.2f} GB")
        print(f"Available memory: {get_available_memory():.2f} GB")

        # Create output directory
        if not os.path.exists(os.path.join(out, 'txts', filename)):
            os.makedirs(os.path.join(out, 'txts', filename))

        # Get page count without loading all pages
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")

        # Determine number of workers based on available memory
        # Assume each page process uses ~0.5GB
        available_mem = get_available_memory()
        max_workers = min(4, int(available_mem / 0.5))  # Cap at 4 workers
        max_workers = max(1, max_workers)  # At least 1 worker

        print(f"Using {max_workers} workers for page processing")

        # Process pages in parallel
        page_args = [(file, i, out, filename) for i in range(num_pages)]

        successful_pages = 0
        with Pool(processes=max_workers) as page_pool:
            for result in tqdm(page_pool.imap_unordered(process_single_page, page_args),
                             total=num_pages, desc=f"Pages in {filename}"):
                if result is not None:
                    successful_pages += 1

                # Monitor memory periodically
                if successful_pages % 50 == 0:
                    current_mem = get_memory_usage()
                    if current_mem > 8.0:  # If using more than 8GB
                        print(f"\nWarning: High memory usage ({current_mem:.2f} GB)")

        print(f"Completed: {filename} ({successful_pages}/{num_pages} pages)")
        return file if successful_pages > 0 else None

    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None


if __name__ == "__main__":
	cores = 1  # Process one document at a time, parallelize pages

	out = "./intermediate/ocr/"

	for path in [out, out + "txts", out + "temp"]:
		os.makedirs(path, exist_ok=True)

	# create a log of complete files
	if not os.path.exists('./intermediate/ocr/complete_files_ocr.log'):
		with open('./intermediate/ocr/complete_files_ocr.log', 'w') as f:
			f.write('')

	# Get PDF files to process
	try:
		files = [os.path.join('./input/pdfs', filename) for filename in os.listdir("./input/pdfs") if filename.lower().endswith('.pdf')]
	except FileNotFoundError:
		print("Error: ./input/pdfs directory not found!")
		print("Make sure you have PDFs in the input/pdfs/ folder")
		sys.exit(1)

	if not files:
		print("No PDF files found in ./input/pdfs/")
		print("Make sure you have PDFs in the input/pdfs/ folder")
		sys.exit(1)

	print(f"Found {len(files)} PDF files to process")


	print("\nChecking CSV and PDF name matching...")
	csv_dir = './input/csv'
	csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

	if not csv_files:
			print(" ERROR: No CSV file found in ./input/csv/")
			print("Please add a CSV file with book metadata")
			sys.exit(1)

	# Read the first CSV file
	csv_path = os.path.join(csv_dir, csv_files[0])
	print(f"Reading CSV: {csv_files[0]}")

	try:
			df = pd.read_csv(csv_path)

			# Get PDF names (without path, with extension)
			pdf_names = set([os.path.basename(f) for f in files])

			# Get CSV filenames - handle both with and without .pdf extension
			csv_filenames = []
			for f in df['filename']:
					f = str(f)
					if not f.lower().endswith('.pdf'):
							f = f + '.pdf'
					csv_filenames.append(f)
			csv_filenames = set(csv_filenames)

			# Check only if PDFs are missing from CSV (not the other way)
			pdfs_not_in_csv = pdf_names - csv_filenames

			if pdfs_not_in_csv:
					print("\n ERROR: Some PDFs are not listed in the CSV!")
					print(f"\n PDFs found but NOT in CSV ({len(pdfs_not_in_csv)}):")
					for pdf in sorted(pdfs_not_in_csv):
							print(f"   - {pdf}")

					print("\n🔧 To fix this:")
					print("   1. Add these PDFs to the CSV 'filename' column")
					print("   2. Or remove these PDFs from input/pdfs/")
					print("\nOCR process ABORTED. All PDFs must be in the CSV.")
					sys.exit(1)

			# Just inform about CSV entries without PDFs (don't fail)
			csv_not_in_pdfs = csv_filenames - pdf_names
			if csv_not_in_pdfs:
					print(f"\n  Note: {len(csv_not_in_pdfs)} CSV entries don't have PDFs yet (this is OK)")

			print(f" All {len(pdf_names)} PDFs are listed in CSV. Proceeding with OCR...")

	except Exception as e:
			print(f" ERROR reading CSV file: {e}")
			print("Make sure the CSV has a 'filename' column")
			sys.exit(1)

	# Filter out already completed files
	with open('./intermediate/ocr/complete_files_ocr.log', 'r') as f:
		complete_files = f.read().split('\n')
	complete_files = [file for file in complete_files if file != '']
	files = [file for file in files if file not in complete_files]

	if not files:
		print("All PDF files have already been processed!")
		sys.exit(0)

	print(f"Processing {len(files)} remaining PDF files...")

	# Setup tesseract path
	is_mac = sys.platform == 'darwin'
	if is_mac:
		pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
	else:
		pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

	# Test tesseract
	try:
		pytesseract.get_tesseract_version()
		print("Tesseract is working")
	except Exception as e:
		print(f"Tesseract not found: {e}")
		print("Install: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)")
		sys.exit(1)

	print(f"Using {cores} CPU cores")
	print(f"Using {cores} document(s) at a time")
	print(f"Initial memory usage: {get_memory_usage():.2f} GB")
	print(f"Available memory: {get_available_memory():.2f} GB")
	print("\nTIP: Use 'htop' or 'top' in another terminal to monitor system resources\n")

	start_time = time.time()
	successful = 0
	failed = 0

	with Manager() as manager:
		counter = manager.Value('i', 0)
		with Pool(processes=cores) as pool:
			for file in tqdm(pool.imap_unordered(process_pdf_file, [(out, file) for file in files]), total=len(files)):
				counter.value += 1
				# Log the complete file path sequentially
				if file is not None:
					successful += 1
					with open('./intermediate/ocr/complete_files_ocr.log', 'a') as f:
						f.write(file + '\n')
				else:
					failed += 1

				gc.collect()

				# Print memory status every 5 documents
				if successful % 5 == 0:
						print(f"\nMemory check - Usage: {get_memory_usage():.2f} GB, Available: {get_available_memory():.2f} GB")

	total_time = time.time() - start_time

	print(f"\nOCR Processing Complete!")
	print(f"Successful: {successful}")
	print(f"Failed: {failed}")
	print(f"Total time: {total_time/60:.1f} minutes")
	print(f"\nOutput locations:")
	print(f"Text files: {out}txts/")