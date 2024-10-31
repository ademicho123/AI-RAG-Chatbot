import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re

class ResumeGenerator:
    def __init__(self):
        # Load the fine-tuned model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained("path_to_finetuned_model")
        self.tokenizer = AutoTokenizer.from_pretrained("path_to_finetuned_model")
        
    def generate_resume(self, job_title):
        # Prepare the prompt
        prompt = f"Generate a professional resume for the position of {job_title}:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=1024,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        # Decode and format the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_resume(generated_text)
    
    def _format_resume(self, text):
        # Clean and format the generated resume
        sections = {
            "Summary": "",
            "Experience": "",
            "Education": "",
            "Skills": "",
            "Certifications": ""
        }
        
        current_section = "Summary"
        lines = text.split("\n")
        
        for line in lines:
            if any(section in line for section in sections.keys()):
                current_section = next(s for s in sections.keys() if s in line)
            else:
                sections[current_section] += line + "\n"
                
        return sections

# Streamlit UI
def main():
    st.title("AI Resume Generator")
    st.write("Generate a professional resume based on job title using AI")
    
    # Input for job title
    job_title = st.text_input("Enter the job title:")
    
    if st.button("Generate Resume"):
        if job_title:
            with st.spinner("Generating resume..."):
                resume_gen = ResumeGenerator()
                resume_sections = resume_gen.generate_resume(job_title)
                
                # Display generated resume
                st.subheader("Generated Resume")
                for section, content in resume_sections.items():
                    if content.strip():
                        st.markdown(f"### {section}")
                        st.write(content)
                
                # Add download button
                st.download_button(
                    label="Download Resume as PDF",
                    data=generate_pdf(resume_sections),
                    file_name="generated_resume.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("Please enter a job title")

def generate_pdf(resume_sections):
    # Placeholder for PDF generation logic
    # You would implement this using a library like FPDF or reportlab
    pass

if __name__ == "__main__":
    main()