from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Books(models.Model):
    """
    Books model for storing book metadata, embeddings, and search data
    """

    # ========================================================================
    # CORE METADATA
    # ========================================================================
    id = models.AutoField(primary_key=True)
    book_id = models.IntegerField(
        null=True,
        blank=True,
        help_text="Unique book identifier"
    )
    title = models.CharField(
        max_length=25500,
        help_text="Book title"
    )
    author = models.CharField(
        max_length=25500,
        help_text="Book author(s)"
    )
    importance = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Importance level or rating"
    )
    topic = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Book topic or subject"
    )

    # ========================================================================
    # GOOGLE BOOKS API DATA
    # ========================================================================
    g_books_title = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Google Books API title"
    )
    g_books_authors = models.JSONField(
        null=True,
        blank=True,
        help_text="Google Books API authors list"
    )
    g_books_publisher = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Publisher from Google Books"
    )
    g_books_published_year = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text="Publication year from Google Books"
    )
    g_books_description = models.TextField(
        null=True,
        blank=True,
        help_text="Book description from Google Books"
    )
    g_books_page_count = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        help_text="Number of pages"
    )
    g_books_categories = models.JSONField(
        null=True,
        blank=True,
        help_text="Book categories from Google Books"
    )
    g_books_average_rating = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(5)],
        help_text="Average rating (0-5)"
    )
    g_books_ratings_count = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Number of ratings"
    )
    g_books_image_links = models.JSONField(
        null=True,
        blank=True,
        help_text="Book cover image links"
    )
    g_books_language = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Book language"
    )
    g_books_preview_link = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Google Books preview link"
    )

    # ========================================================================
    # GOOGLE SCHOLAR DATA
    # ========================================================================
    scholar_title = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Title from Google Scholar"
    )
    scholar_url = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Google Scholar URL"
    )
    scholar_type = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Publication type from Scholar"
    )
    scholar_snippet = models.TextField(
        null=True,
        blank=True,
        help_text="Text snippet from Scholar"
    )
    scholar_publication_info_summary = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Publication info summary"
    )
    scholar_author_names = models.JSONField(
        null=True,
        blank=True,
        help_text="Author names from Scholar"
    )
    scholar_author_links = models.JSONField(
        null=True,
        blank=True,
        help_text="Author profile links from Scholar"
    )

    # ========================================================================
    # GOOGLE SEARCH DATA
    # ========================================================================
    search_title = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Title from Google Search"
    )
    search_url = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="URL from Google Search"
    )
    search_snippet = models.TextField(
        null=True,
        blank=True,
        help_text="Search result snippet"
    )
    search_source = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Search result source"
    )

    # ========================================================================
    # SEARCH-READY DATA
    # ========================================================================
    filename = models.CharField(
        max_length=25500,
        help_text="Original PDF filename"
    )
    public_url = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Public URL for the book"
    )
    public_authors = models.JSONField(
        null=True,
        blank=True,
        help_text="Processed author information"
    )
    public_title = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Processed public title"
    )

    # ========================================================================
    # EMBEDDINGS DATA (CRITICAL FOR SEARCH)
    # ========================================================================
    embed_id_dict = models.JSONField(
        null=True,
        blank=True,
        help_text="Maps chunk_id to text content for embeddings"
    )
    embed_filename = models.CharField(
        max_length=25500,
        null=True,
        blank=True,
        help_text="Reference filename for embeddings"
    )

    # # ========================================================================
    # # METADATA
    # # ========================================================================
    # created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'books'
        verbose_name = 'Book'
        verbose_name_plural = 'Books'
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['author']),
            models.Index(fields=['filename']),
            models.Index(fields=['topic']),
        ]

    def __str__(self):
        return f"{self.title} by {self.author}"

    @property
    def has_embeddings(self):
        """Check if book has embeddings data"""
        return (self.embed_id_dict and
                isinstance(self.embed_id_dict, dict) and
                len(self.embed_id_dict) > 0)

    @property
    def chunk_count(self):
        """Get number of text chunks"""
        if self.has_embeddings:
            return len(self.embed_id_dict)
        return 0

    def get_chunks(self):
        """Get all text chunks as list"""
        if self.has_embeddings:
            return list(self.embed_id_dict.values())
        return []
