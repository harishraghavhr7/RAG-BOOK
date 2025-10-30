import pdfplumber
import os
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from a PDF file using pdfplumber. Returns text with page headers.

    Each page is written with a header like:
      --- Page {n} ---\n
    Args:
        pdf_file_path (str): Path to the PDF file

    Returns:
        str: Extracted text including page headers
    """
    try:
        pages_text = []
        with pdfplumber.open(pdf_file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages_text.append(f"--- Page {page_num} ---\n{text}\n\n")

        return "".join(pages_text)

    except Exception as e:
        logger.error(f"Error extracting text from PDF with pdfplumber: {e}")
        raise


def save_text_to_file(text, output_path):
    """
    Save extracted text to a file.

    Args:
        text (str): Text to save
        output_path (str): Path where to save the text file

    Returns:
        str: Path to the saved file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return output_path
    except Exception as e:
        logger.error(f"Error saving text to file: {e}")
        raise


def process_pdf(pdf_path, output_dir):
    """
    Process a PDF file by extracting its text (page-wise) and saving to a text file.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory where to save the text file

    Returns:
        str: Path to the saved text file
    """
    try:
        # Extract text with page headers
        text = extract_text_from_pdf(pdf_path)

        # Create output filename
        pdf_filename = os.path.basename(pdf_path)
        txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        output_path = os.path.join(output_dir, txt_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save text to file
        save_text_to_file(text, output_path)

        return output_path

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise
