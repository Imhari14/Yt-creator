import os
import re
from fpdf import FPDF
import tempfile
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple, Any

class PDFHandler:
    def __init__(self):
        """Initialize PDF handler with default settings"""
        self.pdf = None
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        
    def _initialize_pdf(self, title="Video Handbook"):
        """Initialize a new PDF with proper formatting"""
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add metadata
        pdf.set_title(title)
        pdf.set_author("Video Learning Assistant")
        pdf.set_creator("Gemini AI")
        
        # Set default font
        pdf.add_font('DejaVu', '', os.path.join(os.path.dirname(__file__), '..', 'fonts', 'DejaVuSansCondensed.ttf'), uni=True)
        pdf.add_font('DejaVu', 'B', os.path.join(os.path.dirname(__file__), '..', 'fonts', 'DejaVuSansCondensed-Bold.ttf'), uni=True)
        pdf.add_font('DejaVu', 'I', os.path.join(os.path.dirname(__file__), '..', 'fonts', 'DejaVuSansCondensed-Oblique.ttf'), uni=True)
        
        pdf.set_font('DejaVu', '', 11)
        
        return pdf
        
    def create_handbook(self, 
                        video_title: str, 
                        content: str, 
                        frames: List[Tuple[np.ndarray, float]] = None, 
                        include_images: bool = True,
                        include_toc: bool = True) -> str:
        """
        Create a comprehensive handbook based on video content
        
        Args:
            video_title: Title of the video
            content: The handbook content in markdown format
            frames: List of video frames to include (optional)
            include_images: Whether to include images in the PDF
            include_toc: Whether to include table of contents
            
        Returns:
            Path to the generated PDF file
        """
        # Initialize PDF
        self.pdf = self._initialize_pdf(video_title)
        
        # Add cover page
        self._add_cover_page(video_title)
        
        # Add table of contents placeholder
        toc_page = None
        if include_toc:
            self.pdf.add_page()
            toc_page = self.pdf.page_no()
            self.pdf.set_font('DejaVu', 'B', 16)
            self.pdf.cell(0, 10, "Table of Contents", ln=True)
            self.pdf.ln(10)
            
            # Add placeholder for TOC that will be filled later
            self.pdf.set_font('DejaVu', '', 11)
            # Placeholder for TOC entries
            self.toc_entries = []
            
        # Parse content and add to PDF
        self._add_content(content, frames if include_images else None)
        
        # If TOC was requested, fill it now that we have all sections
        if include_toc and toc_page is not None:
            self._fill_toc(toc_page)
        
        # Save the PDF to a temporary file
        temp_file = os.path.join(self.temp_dir, f"handbook_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        self.pdf.output(temp_file)
        self.temp_files.append(temp_file)
        
        return temp_file
        
    def _add_cover_page(self, title):
        """Add a cover page to the PDF"""
        self.pdf.add_page()
        
        # Set title
        self.pdf.set_font('DejaVu', 'B', 24)
        self.pdf.ln(60)
        self.pdf.cell(0, 10, title, align='C', ln=True)
        
        # Set subtitle
        self.pdf.set_font('DejaVu', 'I', 16)
        self.pdf.ln(10)
        self.pdf.cell(0, 10, "Comprehensive Handbook", align='C', ln=True)
        
        # Add date
        self.pdf.set_font('DejaVu', '', 12)
        self.pdf.ln(10)
        current_date = datetime.now().strftime("%B %d, %Y")
        self.pdf.cell(0, 10, f"Generated on {current_date}", align='C', ln=True)
        
        # Add page number
        self.pdf.set_font('DejaVu', '', 10)
        self.pdf.ln(120)
        self.pdf.cell(0, 10, "Video Learning Assistant", align='C', ln=True)
        
    def _find_headings(self, content):
        """Find all headings in the content for TOC"""
        heading_pattern = r'^(#{1,3})\s+(.+)$'
        headings = []
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append((level, text))
                
        return headings
    
    def _add_content(self, content, frames=None):
        """Parse markdown content and add to PDF with proper formatting"""
        self.pdf.add_page()
        
        # Track current section for TOC
        current_section = None
        headings = self._find_headings(content)
        heading_pages = {}
        
        # Process content line by line
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle headings
            if line.startswith('# '):
                self.pdf.set_font('DejaVu', 'B', 18)
                text = line[2:].strip()
                if include_toc:
                    heading_pages[text] = self.pdf.page_no()
                    self.toc_entries.append((1, text, self.pdf.page_no()))
                self.pdf.cell(0, 10, text, ln=True)
                self.pdf.ln(5)
                current_section = text
                
            elif line.startswith('## '):
                self.pdf.set_font('DejaVu', 'B', 16)
                text = line[3:].strip()
                if include_toc:
                    self.toc_entries.append((2, text, self.pdf.page_no()))
                self.pdf.cell(0, 10, text, ln=True)
                self.pdf.ln(5)
                
            elif line.startswith('### '):
                self.pdf.set_font('DejaVu', 'B', 14)
                text = line[4:].strip()
                if include_toc:
                    self.toc_entries.append((3, text, self.pdf.page_no()))
                self.pdf.cell(0, 10, text, ln=True)
                self.pdf.ln(5)
                
            # Handle bullet points
            elif line.startswith('- ') or line.startswith('* '):
                self.pdf.set_font('DejaVu', '', 11)
                bullet_text = line[2:].strip()
                self.pdf.cell(5, 6, "", ln=0)
                self.pdf.cell(5, 6, "•", ln=0)
                self.pdf.multi_cell(0, 6, bullet_text)
                self.pdf.ln(2)
                
            # Handle numbered lists
            elif re.match(r'^\d+\.\s', line):
                self.pdf.set_font('DejaVu', '', 11)
                text = re.sub(r'^\d+\.\s', '', line).strip()
                num = re.match(r'^\d+', line).group(0)
                self.pdf.cell(10, 6, f"{num}.", ln=0)
                self.pdf.multi_cell(0, 6, text)
                self.pdf.ln(2)
                
            # Handle code blocks
            elif line.startswith('```'):
                self.pdf.set_font('Courier', '', 10)
                code_content = []
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    code_content.append(lines[i])
                    i += 1
                
                self.pdf.set_fill_color(240, 240, 240)
                for code_line in code_content:
                    self.pdf.cell(0, 6, code_line, ln=True, fill=True)
                self.pdf.ln(5)
                self.pdf.set_font('DejaVu', '', 11)
                
            # Handle images if frames are provided and we're at a good spot to insert one
            elif frames and (line == '' or i % 20 == 0) and len(frames) > 0:
                # Add image near relevant content
                frame, timestamp = frames.pop(0)
                
                # Save frame to temp file
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_path = os.path.join(self.temp_dir, f"frame_{timestamp}.jpg")
                img.save(img_path)
                self.temp_files.append(img_path)
                
                # Add image to PDF
                self.pdf.ln(5)
                img_width = min(180, self.pdf.w - 40)  # Max width with margins
                img_height = img_width * img.height / img.width
                
                # Center the image
                x = (self.pdf.w - img_width) / 2
                self.pdf.image(img_path, x=x, w=img_width)
                
                # Add caption with timestamp
                minutes, seconds = divmod(int(timestamp), 60)
                caption = f"Frame at {minutes:02d}:{seconds:02d}"
                self.pdf.set_font('DejaVu', 'I', 10)
                self.pdf.cell(0, 10, caption, align='C', ln=True)
                self.pdf.set_font('DejaVu', '', 11)
                self.pdf.ln(5)
                
            # Regular paragraph text
            elif line != '':
                self.pdf.set_font('DejaVu', '', 11)
                self.pdf.multi_cell(0, 6, line)
                self.pdf.ln(2)
                
            i += 1
            
    def _fill_toc(self, toc_page):
        """Fill the table of contents with collected entries"""
        # Store current page and go back to TOC page
        current_page = self.pdf.page_no()
        self.pdf.page = toc_page
        
        for level, text, page in self.toc_entries:
            # Format based on heading level
            if level == 1:
                self.pdf.set_font('DejaVu', 'B', 12)
                indent = 0
            elif level == 2:
                self.pdf.set_font('DejaVu', '', 11)
                indent = 10
            else:
                self.pdf.set_font('DejaVu', '', 10)
                indent = 20
                
            # Add dot leaders between text and page number
            self.pdf.cell(indent, 6, '', ln=0)
            
            # Calculate width for text and dots
            text_width = self.pdf.get_string_width(text)
            page_num_width = self.pdf.get_string_width(str(page))
            available_width = self.pdf.w - 20 - indent - page_num_width
            
            # Ensure we don't overflow with text
            if text_width > available_width - 20:  # Leave space for dots
                while text_width > available_width - 20:
                    text = text[:-1]
                    text_width = self.pdf.get_string_width(text)
                text += "..."
                
            self.pdf.cell(available_width, 6, text, ln=0)
            
            # Add page number
            self.pdf.cell(page_num_width, 6, str(page), ln=1, align='R')
            
            # Add some spacing between main sections
            if level == 1:
                self.pdf.ln(2)
        
        # Return to the page we were on
        self.pdf.page = current_page
    
    def get_pdf_download_link(self, pdf_path):
        """Generate a download link for the PDF"""
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        filename = os.path.basename(pdf_path)
        
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download PDF Handbook</a>'
        return href
    
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass