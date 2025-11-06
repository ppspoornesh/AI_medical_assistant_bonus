from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import pandas as pd
from loguru import logger

class ReportAssemblyAgent:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        logger.info("ReportAssemblyAgent ready – using ReportLab for PDF generation")

    def assemble_report(self, sections):
        """Builds professional PDF with headings, text, tables, and images."""
        logger.info(f"Assembling report with {len(sections)} section(s)")
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=60)
        elements = []

        for title, content in sections.items():
            elements.append(Paragraph(title, self.styles['Heading1']))
            elements.append(Spacer(1, 12))

            if isinstance(content, str) and '|' in content and content.startswith('|'):
                # Markdown table → pandas → ReportLab Table
                try:
                    df = pd.read_markdown(io.StringIO(content))
                    data = [df.columns.tolist()] + df.values.tolist()
                    table = Table(data)
                    table.setStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ])
                    elements.append(table)
                except:
                    elements.append(Paragraph(content, self.styles['Normal']))
            elif isinstance(content, dict) and "path" in content:
                # Insert image
                try:
                    img = RLImage(content["path"], width=450, height=250)
                    elements.append(img)
                    if content.get("text"):
                        elements.append(Paragraph(content["text"], self.styles['Italic']))
                except:
                    elements.append(Paragraph("[Image failed to load]", self.styles['Normal']))
            else:
                elements.append(Paragraph(content, self.styles['Normal']))

            elements.append(Spacer(1, 24))

        doc.build(elements)
        buffer.seek(0)
        logger.success("PDF report assembled and ready for download")
        return buffer