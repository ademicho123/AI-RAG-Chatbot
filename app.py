import streamlit as st
from resume_model import ResumeGenerator
import pypdf
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from dotenv import load_dotenv
import os

def create_pdf(resume_sections):
    """Create a PDF from the resume sections"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom style for headers
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor('#2c3e50')
    ))
    
    # Build the PDF content
    content = []
    for section, text in resume_sections.items():
        # Add section header
        content.append(Paragraph(section, styles['SectionHeader']))
        # Add section content
        paragraphs = text.split('\n')
        for para in paragraphs:
            if para.strip():
                content.append(Paragraph(para, styles['Normal']))
        content.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

def main():
    # Load environment variables
    load_dotenv()
    
    st.set_page_config(page_title="AI Resume Generator", layout="wide")
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .section-header {
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 AI Resume Generator")
    
    # Initialize session state for the resume generator
    if 'resume_gen' not in st.session_state:
        with st.spinner("Loading AI model... This may take a minute..."):
            try:
                st.session_state.resume_gen = ResumeGenerator()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please make sure you have a valid HUGGINGFACE_TOKEN in your .env file")
                st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_title = st.text_input("Enter the job title:", placeholder="e.g., Senior Software Engineer")
        additional_info = st.text_area(
            "Additional information (optional):",
            placeholder="Enter any specific requirements or preferences...",
            height=100
        )
    
    with col2:
        st.markdown("### Tips for best results:")
        st.markdown("""
        - Be specific with the job title
        - Include key skills in additional info
        - Mention preferred experience level
        - Add industry-specific requirements
        """)
    
    if st.button("🚀 Generate Resume", type="primary"):
        if job_title:
            with st.spinner("Generating your professional resume..."):
                try:
                    resume_sections = st.session_state.resume_gen.generate_resume(
                        job_title,
                        additional_info
                    )
                    
                    # Display generated resume
                    st.markdown("## Generated Resume")
                    
                    for section, content in resume_sections.items():
                        if content.strip():
                            st.markdown(f"<div class='section-header'>{section}</div>", 
                                    unsafe_allow_html=True)
                            st.write(content)
                            st.markdown("---")
                    
                    # Create PDF
                    pdf_buffer = create_pdf(resume_sections)
                    
                    # Add download button
                    st.download_button(
                        label="📄 Download Resume as PDF",
                        data=pdf_buffer,
                        file_name=f"resume_{job_title.lower().replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred while generating the resume: {str(e)}")
        else:
            st.error("Please enter a job title")

if __name__ == "__main__":
    main()