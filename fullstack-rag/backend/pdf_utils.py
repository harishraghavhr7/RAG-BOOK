from PyPDF2 import PdfReader
import os
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Create PDF reader object
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            # Extract text from each page
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            
            # Join all pages with double newlines
            return '\n\n'.join(text)
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
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
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        return output_path
    except Exception as e:
        logger.error(f"Error saving text to file: {str(e)}")
        raise

def process_pdf(pdf_path, output_dir):
    """
    Process a PDF file by extracting its text and saving to a text file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory where to save the text file
        
    Returns:
        str: Path to the saved text file
    """
    try:
        # Extract text from PDF
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
        logger.error(f"Error processing PDF: {str(e)}")
        raise
