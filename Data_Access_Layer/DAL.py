from typing import List, Tuple, Any
from llama_parse import LlamaParse
from llama_index.core import Document
import os
import tempfile
import logging
from pathlib import Path

class DataLoader:
    """
    Class to load and extract structured Markdown content from PDF files using LlamaParse.
    Output is saved as .md files and returned as a list of LlamaIndex Document objects.
    """

    def extract_text_from_pdf(self, pdf_files: List[Any], output_dir: str = ".") -> Tuple[List[Document], List[str]]:
        """
        Extract text from PDF files, save as Markdown, and return Document objects.

        Args:
            pdf_files: List of file-like objects with name and read() method.
            output_dir: Directory to save Markdown files (default: current directory).

        Returns:
            Tuple of (list of Document objects, list of error messages).
        """
        logging.basicConfig(level=logging.INFO)
        documents = []
        errors = []
        api_key = os.getenv("LLAMAPARSE_API_KEY")
        if not api_key:
            errors.append("LLAMAPARSE_API_KEY environment variable not set")
            return documents, errors

        parser = LlamaParse(api_key=api_key, result_type="markdown")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(pdf_files, list):
            errors.append("pdf_files must be a list")
            return documents, errors

        for pdf_file in pdf_files:
            try:
                if not hasattr(pdf_file, 'name') or not pdf_file.name.lower().endswith(".pdf"):
                    errors.append(f"Skipping invalid file: {getattr(pdf_file, 'name', 'unknown')}")
                    continue

                logging.info(f"Processing file: {pdf_file.name}")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                    temp_pdf.write(pdf_file.read())
                    temp_pdf_path = temp_pdf.name

                try:
                    parsed_docs = parser.load_data(temp_pdf_path)
                    if not parsed_docs:
                        errors.append(f"No content extracted from {pdf_file.name}")
                        continue

                    md_file = output_path / f"{Path(pdf_file.name).stem}.md"
                    if md_file.exists():
                        errors.append(f"Markdown file {md_file} already exists, skipping write")
                    else:
                        with open(md_file, "w", encoding="utf-8") as f:
                            f.write(parsed_docs[0].text)
                    documents.extend(parsed_docs)

                finally:
                    os.unlink(temp_pdf_path)

            except FileNotFoundError as e:
                errors.append(f"File access error for {pdf_file.name}: {str(e)}")
            except IndexError as e:
                errors.append(f"Parsing error for {pdf_file.name}: No content parsed")
            except Exception as e:
                errors.append(f"Error processing {pdf_file.name}: {str(e)}")

        return documents, errors