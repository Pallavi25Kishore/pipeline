from django.shortcuts import render
from django.http import JsonResponse
from .models import Books

# Create your views here.

def books_stats(request):
    """API endpoint for books statistics"""
    total_books = Books.objects.count()
    books_with_embeddings = Books.objects.filter(
        embed_id_dict__isnull=False
    ).exclude(embed_id_dict__exact={}).count()

    return JsonResponse({
        'total_books': total_books,
        'books_with_embeddings': books_with_embeddings,
        'embedding_percentage': (books_with_embeddings / total_books * 100) if total_books > 0 else 0
    })
