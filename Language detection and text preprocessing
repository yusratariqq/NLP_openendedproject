"""
Language detection and text preprocessing utilities
"""
from langdetect import detect, LangDetectException
import re

def detect_language(text: str) -> str:
    """
    Detect language of input text

    Args:
        text: Input text

    Returns:
        Language code ('ar', 'ta', or 'unknown')
    """
    try:
        lang = detect(text)
        # Map detected language to our supported languages
        if lang in ['ar', 'ta']:
            return lang
        return 'unknown'
    except LangDetectException:
        return 'unknown'

def preprocess_text(text: str, lang: str = None) -> str:
    """
    Preprocess text for embedding

    Args:
        text: Input text
        lang: Language code (optional)

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep language-specific characters
    # Keep Urdu/Tamil scripts
    text = text.strip()

    # Truncate if too long (optional)
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]

    return text

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks

    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks

def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncate text to maximum characters

    Args:
        text: Input text
        max_chars: Maximum characters

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
