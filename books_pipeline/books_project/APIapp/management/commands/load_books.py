from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from APIapp.models import Books
import os
import json
from tqdm import tqdm
import logging
import csv
from numpy import nan
import ast
import pandas as pd
import sys
from django.conf import settings

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Load books data from CSV file with embeddings into the Database'

    def add_arguments(self, parser):
        parser.add_argument(
            'csv_file',
            type=str,
            nargs='?',  # Optional argument
            help='CSV file location (if not provided, will search for latest)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=getattr(settings, 'BOOKS_PIPELINE', {}).get('BATCH_SIZE', 100),
            help='Batch size for database operations (default: 100)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reload even if data exists'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Test run without saving to database'
        )

    def find_latest_csv(self):
        """Find the latest CSV file with embeddings"""
        # Search paths from settings or default
        search_paths = getattr(settings, 'BOOKS_PIPELINE', {}).get('CSV_SEARCH_PATHS', [
            '../input/csv',
            '../../input/csv',
            'input/csv',
            os.path.join(os.path.dirname(settings.BASE_DIR), 'input', 'csv')
        ])

        for csv_dir in search_paths:
            if os.path.exists(csv_dir):
                csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

                if csv_files:
                    # Prefer files with 'embeddings' in the name
                    embedding_csvs = [f for f in csv_files if 'embedding' in f.lower()]

                    if embedding_csvs:
                        embedding_csvs.sort()
                        latest_csv = embedding_csvs[-1]
                    else:
                        csv_files.sort()
                        latest_csv = csv_files[-1]

                    csv_path = os.path.join(csv_dir, latest_csv)
                    self.stdout.write(
                        self.style.SUCCESS(f"Found CSV file: {csv_path}")
                    )
                    return csv_path

        self.stdout.write(
            self.style.ERROR("No CSV files found in search paths:")
        )
        for path in search_paths:
            self.stdout.write(f"  - {path}")
        return None

    def validate_and_clean_data(self, row):
        """Validate and clean data for database insertion"""
        cleaned_row = {}

        # Data mapping from CSV columns to model fields
        data_to_model_mapping = {
            # Core metadata
            "ID": "book_id",
            "Title": "title",
            "Author": "author",
            "Importance": "importance",
            "Topic": "topic",

            # Google Books API data
            "title": "g_books_title",
            "authors": "g_books_authors",
            "publisher": "g_books_publisher",
            "publishedDate": "g_books_published_year",
            "description": "g_books_description",
            "pageCount": "g_books_page_count",
            "categories": "g_books_categories",
            "averageRating": "g_books_average_rating",
            "ratingsCount": "g_books_ratings_count",
            "imageLinks": "g_books_image_links",
            "language": "g_books_language",
            "previewLink": "g_books_preview_link",

            # Google Scholar data
            "scholar_title": "scholar_title",
            "scholar_url": "scholar_url",
            "scholar_type": "scholar_type",
            "scholar_snippet": "scholar_snippet",
            "scholar_publication_info_summary": "scholar_publication_info_summary",
            "scholar_author_names": "scholar_author_names",
            "scholar_author_links": "scholar_author_links",

            # Google Search data
            "search_title": "search_title",
            "search_url": "search_url",
            "search_snippet": "search_snippet",
            "search_source": "search_source",

            # Search-ready data
            "filename": "filename",
            "public_url": "public_url",
            "public_authors": "public_authors",
            "public_title": "public_title",
            "embed_id_dict": "embed_id_dict",
            "embed_filename": "embed_filename",
        }

        for csv_key, model_field in data_to_model_mapping.items():
            value = row.get(csv_key)

            # Handle None, empty, or NaN values
            if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
                value = None

            # Special handling for specific fields
            elif csv_key == 'publishedDate':
                value = str(value) if value else None

            elif csv_key == 'ID':
                try:
                    value = int(float(value)) if value else None
                except (ValueError, TypeError):
                    value = None

            # Handle JSON fields
            elif model_field in ['g_books_authors', 'g_books_categories', 'g_books_image_links',
                               'scholar_author_names', 'scholar_author_links', 'public_authors',
                               'embed_id_dict']:
                if isinstance(value, str) and value:
                    try:
                        # Try JSON first
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            # Try ast.literal_eval
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If it's embed_id_dict and starts with {, try eval as last resort
                            if model_field == 'embed_id_dict' and value.startswith('{'):
                                try:
                                    value = eval(value)
                                except Exception as e:
                                    logger.warning(f"Could not parse embed_id_dict: {e}")
                                    value = {}
                            else:
                                value = None
                elif not isinstance(value, (dict, list)):
                    value = None

            # Handle float fields
            elif model_field in ['g_books_page_count', 'g_books_average_rating', 'g_books_ratings_count']:
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = None
                elif isinstance(value, (int, float)):
                    value = float(value)

            # Handle string fields - truncate if too long
            elif isinstance(value, str):
                if len(value) > 25500:
                    value = value[:25500]
                    logger.warning(f"Truncated long field {model_field}")

            cleaned_row[model_field] = value

        return cleaned_row

    def handle(self, *args, **kwargs):
        csv_file = kwargs.get('csv_file')
        batch_size = kwargs.get('batch_size')
        force = kwargs.get('force', False)
        dry_run = kwargs.get('dry_run', False)

        self.stdout.write(
            self.style.SUCCESS('=== Books Database Loading Command ===')
        )

        # Find CSV file if not provided
        if not csv_file:
            csv_file = self.find_latest_csv()
            if not csv_file:
                raise CommandError('No CSV file found')

        # Check if file exists
        if not os.path.exists(csv_file):
            raise CommandError(f'CSV file not found: {csv_file}')

        # Check if data already exists (unless force is used)
        if not force and not dry_run:
            existing_books = Books.objects.count()
            if existing_books > 0:
                self.stdout.write(
                    self.style.WARNING(f'Database already contains {existing_books} books.')
                )
                self.stdout.write('Use --force to reload data anyway.')
                return

        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No data will be saved')
            )

        logger.info('Starting books data loading process')
        self.stdout.write(f'Loading books data from: {csv_file}')

        books_list = []
        successful_count = 0
        error_count = 0

        try:
            # Read CSV with pandas for better handling
            self.stdout.write('Reading CSV file...')
            df = pd.read_csv(csv_file)
            self.stdout.write(
                self.style.SUCCESS(f"Found {len(df)} rows in CSV")
            )

            # Check for embed_id_dict column
            if 'embed_id_dict' in df.columns:
                books_with_embeddings = df[
                    df['embed_id_dict'].notna() &
                    (df['embed_id_dict'] != '') &
                    (df['embed_id_dict'] != '{}')
                ]
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Books with embeddings: {len(books_with_embeddings)}/{len(df)}"
                    )
                )
            else:
                self.stdout.write(
                    self.style.WARNING("Warning: embed_id_dict column not found")
                )

            # Process each row with transaction
            self.stdout.write('Processing books...')

            with transaction.atomic():
                for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing books"):
                    try:
                        # Clean and validate data
                        cleaned_data = self.validate_and_clean_data(row.to_dict())

                        # Create Books instance
                        book = Books(**cleaned_data)
                        books_list.append(book)

                        # Batch insert
                        if len(books_list) >= batch_size:
                            if not dry_run:
                                Books.objects.bulk_create(books_list, ignore_conflicts=True)
                            successful_count += len(books_list)
                            logger.info(f'Batch processed. Total: {successful_count}')
                            self.stdout.write(
                                f'Processed batch: {len(books_list)} books (Total: {successful_count})'
                            )
                            books_list = []  # reset the list for the next batch

                    except Exception as e:
                        error_count += 1
                        logger.error(f'Error processing row {index}: {e}')
                        if error_count <= 5:  # Only show first 5 errors
                            self.stdout.write(
                                self.style.ERROR(f"Error processing row {index}: {e}")
                            )

                # Insert remaining books
                if books_list:
                    if not dry_run:
                        Books.objects.bulk_create(books_list, ignore_conflicts=True)
                    successful_count += len(books_list)
                    logger.info('Final batch processed.')
                    self.stdout.write(f'Processed final batch: {len(books_list)} books')

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error reading CSV file: {e}')
            )
            logger.error(f'Error reading CSV file: {e}')
            raise CommandError(f'CSV processing failed: {e}')

        # Final summary
        self.stdout.write('')
        if dry_run:
            self.stdout.write(
                self.style.SUCCESS('DRY RUN COMPLETE - No data was saved')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS('Database loading complete!')
            )

        self.stdout.write(f'Successfully processed: {successful_count} books')
        if error_count > 0:
            self.stdout.write(
                self.style.WARNING(f'Errors encountered: {error_count}')
            )

        # Verify database contents (skip in dry run)
        if not dry_run:
            try:
                total_books = Books.objects.count()
                books_with_embeddings = Books.objects.filter(
                    embed_id_dict__isnull=False
                ).exclude(embed_id_dict__exact={}).count()

                self.stdout.write('')
                self.stdout.write('=== Database Verification ===')
                self.stdout.write(f'Total books in database: {total_books}')
                self.stdout.write(f'Books with embeddings: {books_with_embeddings}')

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error verifying database: {e}')
                )

        logger.info('Books data loading process completed')
        self.stdout.write(
            self.style.SUCCESS('=== Loading Process Complete ===')
        )
