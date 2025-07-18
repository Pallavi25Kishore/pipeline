from django.contrib import admin
from django.utils.html import format_html
from .models import Books

@admin.register(Books)
class BooksAdmin(admin.ModelAdmin):
    """
    Django Admin configuration for Books model
    """

    # List view configuration
    list_display = [
        'title',
        'author',
        'importance',
        'topic',
        'filename',
        'has_embeddings_display',
        'chunk_count_display',
        # 'created_at'
    ]

    list_filter = [
        'importance',
        'topic',
        'g_books_language',
        # 'created_at',
        # 'updated_at'
    ]

    search_fields = [
        'title',
        'author',
        'filename',
        'g_books_title',
        'scholar_title',
        'search_title'
    ]

    readonly_fields = [
        # 'created_at',
        # 'updated_at',
        'has_embeddings_display',
        'chunk_count_display'
    ]

    list_per_page = 25

    # Detailed view configuration
    fieldsets = (
        ('Core Information', {
            'fields': (
                'book_id',
                'title',
                'author',
                'importance',
                'topic',
                'filename'
            )
        }),
        ('Google Books Data', {
            'fields': (
                'g_books_title',
                'g_books_authors',
                'g_books_publisher',
                'g_books_published_year',
                'g_books_description',
                'g_books_page_count',
                'g_books_categories',
                'g_books_average_rating',
                'g_books_ratings_count',
                'g_books_image_links',
                'g_books_language',
                'g_books_preview_link'
            ),
            'classes': ('collapse',)
        }),
        ('Scholar Data', {
            'fields': (
                'scholar_title',
                'scholar_url',
                'scholar_type',
                'scholar_snippet',
                'scholar_publication_info_summary',
                'scholar_author_names',
                'scholar_author_links'
            ),
            'classes': ('collapse',)
        }),
        ('Search Data', {
            'fields': (
                'search_title',
                'search_url',
                'search_snippet',
                'search_source'
            ),
            'classes': ('collapse',)
        }),
        ('Public Data', {
            'fields': (
                'public_url',
                'public_authors',
                'public_title'
            ),
            'classes': ('collapse',)
        }),
        ('Embeddings', {
            'fields': (
                'has_embeddings_display',
                'chunk_count_display',
                'embed_filename'
            ),
        }),
        # ('Metadata', {
        #     'fields': (
        #         'created_at',
        #         'updated_at'
        #     ),
        #     'classes': ('collapse',)
        # }),
    )

    def has_embeddings_display(self, obj):
        """Display if book has embeddings"""
        if obj.has_embeddings:
            return format_html(
                '<span style="color: green;">✓ Yes</span>'
            )
        return format_html(
            '<span style="color: red;">✗ No</span>'
        )
    has_embeddings_display.short_description = 'Has Embeddings'

    def chunk_count_display(self, obj):
        """Display number of chunks"""
        count = obj.chunk_count
        if count > 0:
            return format_html(
                '<span style="color: blue;">{} chunks</span>', count
            )
        return format_html(
            '<span style="color: gray;">0 chunks</span>'
        )
    chunk_count_display.short_description = 'Chunks'

    # Custom actions
    actions = ['export_selected_books']

    def export_selected_books(self, request, queryset):
        """Export selected books to CSV"""
        # This could be implemented to export data
        self.message_user(request, f"Export functionality to be implemented for {queryset.count()} books")
    export_selected_books.short_description = "Export selected books"
