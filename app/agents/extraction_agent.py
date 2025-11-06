from pypdf import PdfReader
from docx import Document as DocxDocument
import pandas as pd
from PIL import Image
import pytesseract
import re
from loguru import logger

class ExtractionAgent:
    def extract_text(self, path, section=None):
        """Extracts exact text from a section – no summarization unless requested."""
        logger.info(f"Extracting text from {path} | Section: {section or 'Full'}")
        text = ""

        if path.endswith('.pdf'):
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        elif path.endswith('.docx'):
            doc = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        elif path.endswith('.xlsx'):
            df_dict = pd.read_excel(path, sheet_name=None)
            text = "\n".join(sheet.to_string() for sheet in df_dict.values())

        if section:
            # Smart regex: finds section header and captures until next major header
            pattern = re.compile(
                r'(' + re.escape(section) + r'[\s\S]*?)(?=(?:\n[A-Z ]{5,}\n)|$)', 
                re.IGNORECASE
            )
            match = pattern.search(text)
            extracted = match.group(1).strip() if match else text
            logger.debug(f"Section '{section}' extracted ({len(extracted)} chars)")
            return extracted

        return text

    def extract_table(self, path, page_num=1, sheet_name=None):
        """Extracts table as Markdown – preserves structure exactly."""
        logger.info(f"Extracting table from {path} | Page/Sheet: {page_num or sheet_name}")
        if path.endswith('.pdf'):
            reader = PdfReader(path)
            page = reader.pages[page_num - 1]
            lines = [line.split() for line in page.extract_text().split('\n') if line.strip()]
            df = pd.DataFrame(lines[1:], columns=lines[0])  # Assume first row is header
            return df.to_markdown(index=False)
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path, sheet_name=sheet_name or 0)
            return df.to_markdown(index=False)
        return ""

    def extract_image(self, path):
        """Returns image path + OCR text – for embedding in report."""
        logger.info(f"Extracting image: {path}")
        if path.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            return {"text": text, "path": path}
        return {}


if __name__ == "__main__":
    extractor = ExtractionAgent()
    print(extractor.extract_text("docs/sample_dataset/cmh-2022-0365.pdf", "Introduction"))
    print(extractor.extract_table("docs/sample_dataset/cmh-2022-0365.pdf", 1))