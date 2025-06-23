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

def process_pdf_file(args):
	try:
		out, file = args
		this_lang = 'eng'

		print(file)
		timing_info = {"cv": 0, "tesseract": 0, "compression": 0}
		t0 = time.time()

		pages = convert_from_path(file)  # PDF to PIL images
		pages = [np.array(page.convert('RGB')) for page in pages]  # PIL to numpy => BGR to RGB
		pages = [denoise(page) for page in pages]  # Denoise
		pages = [sharpen(page) for page in pages]  # Sharpen

		timing_info["cv"] = time.time()

		# Get filename for outputs
		filename = os.path.basename(file).split(".")[0]

		# Create output directory for text files
		if not os.path.exists(os.path.join(out, 'txts', filename)):
			os.makedirs(os.path.join(out,'txts', filename))

		# Text extraction only
		for i, page in enumerate(pages):
			try:
				txt_content = pytesseract.image_to_string(page, lang=this_lang, config='--psm 6')
			except Exception as e:
				txt_content = ""
				print(f"\n\nTesseract img2str failed on PAGE {i} of {file}: {e}")

			# Save individual text file
			with open(os.path.join(out, "txts", filename, f'page_{i}.txt'), 'w', encoding='utf-8') as f:
				f.write(txt_content)

		del pages
		timing_info["tesseract"] = time.time()

		print(f"Completed: {filename} ({i+1} pages)")
		return file

	except Exception as e:
		print(f"Error processing {file}: {e}")
		return None


if __name__ == "__main__":
	cores = os.cpu_count() // 2

	out = "./intermediate/ocr/"

	for path in [out, out + "txts", out + "temp"]:
		os.makedirs(path, exist_ok=True)

	# create a log of complete files
	if not os.path.exists('./intermediate/ocr/complete_files_ocr.log'):
		with open('./intermediate/ocr/complete_files_ocr.log', 'w') as f:
			f.write('')

	# Get PDF files to process
	try:
		files = [os.path.join('./input/pdfs', filename) for filename in os.listdir("./input/pdfs") if filename.endswith('.pdf')]
	except FileNotFoundError:
		print("Error: ./input/pdfs directory not found!")
		print("Make sure you have PDFs in the input/pdfs/ folder")
		sys.exit(1)

	if not files:
		print("No PDF files found in ./input/pdfs/")
		print("Make sure you have PDFs in the input/pdfs/ folder")
		sys.exit(1)

	print(f"Found {len(files)} PDF files to process")

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

	total_time = time.time() - start_time

	print(f"\nOCR Processing Complete!")
	print(f"Successful: {successful}")
	print(f"Failed: {failed}")
	print(f"Total time: {total_time/60:.1f} minutes")
	print(f"\nOutput locations:")
	print(f"Text files: {out}txts/")