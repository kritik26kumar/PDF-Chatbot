import fitz  # PyMuPDF
from PIL import Image
import pytesseract.pytesseract as pt
import streamlit as st
import tempfile
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.documents.elements import NarrativeText, Table, Image
import os

class data_loader:
    """Class to load and extract data from user-uploaded files."""
    def extract_text_from_pdf(self, pdf_files):
        """Extract text, tables, and images from PDF files."""
        extracted_content = []
        if not pdf_files:
            st.error("No PDF files provided.")
            return extracted_content

        for pdf_file in pdf_files:
            if not pdf_file.name.lower().endswith('.pdf'):
                st.error(f"File {pdf_file.name} is not a valid PDF.")
                continue

            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Use Unstructured.io for advanced extraction
                elements = partition_pdf(tmp_file_path, strategy="hi_res", ocr_languages="eng")
                for element in elements:
                    if isinstance(element, NarrativeText):
                        extracted_content.append({"type": "text", "content": element.text, "source": pdf_file.name})
                    elif isinstance(element, Table):
                        extracted_content.append({"type": "table", "content": element.text, "source": pdf_file.name})
                    elif isinstance(element, Image):
                        extracted_content.append({"type": "image", "content": element.text or "Image content", "source": pdf_file.name})

                # Fallback to PyMuPDF for text if Unstructured.io fails
                if not extracted_content:
                    text = ""
                    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
                        for page_num, page in enumerate(doc, 1):
                            try:
                                extracted_text = page.get_text()
                                if not extracted_text.strip():
                                    pix = page.get_pixmap()
                                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                    extracted_text = pt.image_to_string(image)
                                text += extracted_text + "\n"
                            except Exception as e:
                                st.warning(f"Error processing page {page_num} in {pdf_file.name}: {str(e)}")
                    if text.strip():
                        extracted_content.append({"type": "text", "content": text, "source": pdf_file.name})
            except fitz.FileDataError:
                st.error(f"File {pdf_file.name} is corrupted or not a valid PDF.")
            except Exception as e:
                st.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_file_path)

        if not extracted_content:
            st.warning("No content could be extracted from the provided PDFs.")
        return extracted_content

    def extract_text_from_image(self, image_file):
        """Extract text from an image file using Tesseract OCR."""
        extracted_content = []
        if not image_file:
            st.error("No image file provided.")
            return extracted_content

        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        if not image_file.name.lower().endswith(valid_extensions):
            st.error(f"File {image_file.name} is not a supported image format ({', '.join(valid_extensions)}).")
            return extracted_content

        try:
            image = Image.open(image_file)
            text = pt.image_to_string(image)
            if text.strip():
                extracted_content.append({"type": "text", "content": text, "source": image_file.name})
            else:
                st.warning(f"No text could be extracted from {image_file.name}.")
        except pt.TesseractError as e:
            st.error(f"Tesseract OCR error for {image_file.name}: {str(e)}")
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {str(e)}")
        
        return extracted_content

    def extract_content_from_excel(self, excel_files):
        """Extract content from Excel files using Unstructured.io."""
        extracted_content = []
        if not excel_files:
            st.error("No Excel files provided.")
            return extracted_content

        for excel_file in excel_files:
            if not excel_file.name.lower().endswith(('.xlsx', '.xls')):
                st.error(f"File {excel_file.name} is not a valid Excel file.")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(excel_file.read())
                tmp_file_path = tmp_file.name

            try:
                elements = partition_xlsx(tmp_file_path)
                for element in elements:
                    if isinstance(element, NarrativeText):
                        extracted_content.append({"type": "text", "content": element.text, "source": excel_file.name})
                    elif isinstance(element, Table):
                        extracted_content.append({"type": "table", "content": element.text, "source": excel_file.name})
            except Exception as e:
                st.error(f"Error processing Excel {excel_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_file_path)

        if not extracted_content:
            st.warning("No content could be extracted from the provided Excel files.")
        return extracted_content

    def load_files(self, files):
        """Process multiple files of different types."""
        pdf_files = [f for f in files if f.name.lower().endswith('.pdf')]
        image_files = [f for f in files if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        excel_files = [f for f in files if f.name.lower().endswith(('.xlsx', '.xls'))]

        extracted_content = []
        extracted_content.extend(self.extract_text_from_pdf(pdf_files))
        extracted_content.extend(self.extract_text_from_image(image_files[0] if image_files else None))
        extracted_content.extend(self.extract_content_from_excel(excel_files))
        
        return extracted_content