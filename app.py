import streamlit as st
import logging
from resume_model import generate_section, generate_resume
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
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
        .error-box {
            background-color: #ffeeee;
            border: 1px solid #ff0000;
            color: #ff0000;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 AI Resume Generator")
    
    # Input columns
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
                    # Call the generate_resume function
                    resume_sections = generate_resume(job_title, additional_info)
                    
                    if not resume_sections:
                        st.markdown("""
                        <div class="error-box">
                        <strong>No resume sections were generated.</strong><br>
                        Possible reasons:
                        <ul>
                        <li>API connection issue</li>
                        <li>Invalid job title</li>
                        <li>Temporary service disruption</li>
                        </ul>
                        Please check your API configuration and try again.
                        </div>
                        """, unsafe_allow_html=True)
                        # Log the failure
                        logger.warning(f"No resume sections generated for job title: {job_title}")
                        return
                    
                    # Display the generated resume
                    st.markdown("## Generated Resume")
                    for section, content in resume_sections.items():
                        if content.strip():  # Ensure non-empty content
                            st.markdown(f"<div class='section-header'>{section}</div>", 
                                        unsafe_allow_html=True)
                            st.write(content)
                            st.markdown("---")
                except Exception as e:
                    # More detailed error handling
                    error_message = str(e)
                    st.markdown(f"""
                    <div class="error-box">
                    <strong>An error occurred while generating the resume:</strong><br>
                    {error_message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Log the full error for debugging
                    logger.error(f"Resume generation error: {error_message}", exc_info=True)
        else:
            st.error("Please enter a job title")

if __name__ == "__main__":
    main()