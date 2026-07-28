from fpdf import FPDF

def markdown_to_pdf_bytes(markdown_text: str, title: str = "Career Assessment Report") -> bytes:
    """
    Render a markdown report into a downloadable PDF safely using fpdf2.
    Dynamically adjusts cell widths to prevent horizontal space overflow errors.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Header (0 width forces auto full page width)
    pdf.set_font("Helvetica", "B", 16)
    clean_title = title.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, clean_title)
    pdf.ln(4)
    
    # Set base font for body
    pdf.set_font("Helvetica", size=10)
    
    for raw_line in markdown_text.split("\n"):
        line = raw_line.rstrip()
        
        # Handle empty lines
        if not line.strip():
            pdf.ln(3)
            continue
            
        # Safe character encoding to prevent Unicode/Emoji crashes
        clean_line = line.encode('latin-1', 'replace').decode('latin-1')
        
        # Formatting Markdown Headings
        if clean_line.startswith("#"):
            heading_level = len(clean_line) - len(clean_line.lstrip("#"))
            clean_heading = clean_line.lstrip("#").strip()
            
            if heading_level == 1:
                pdf.set_font("Helvetica", "B", 14)
                pdf.ln(2)
            elif heading_level == 2:
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(2)
            else:
                pdf.set_font("Helvetica", "B", 10)
                
            pdf.multi_cell(0, 8, clean_heading)
            pdf.set_font("Helvetica", size=10)
        
        # Formatting Bullet Points
        elif clean_line.startswith("* ") or clean_line.startswith("- "):
            bullet_text = "• " + clean_line[2:].strip()
            pdf.multi_cell(0, 6, bullet_text)
            
        # Regular Paragraph Text
        else:
            pdf.multi_cell(0, 6, clean_line)
            
    # Return output as raw bytes for Streamlit download button
    return bytes(pdf.output())