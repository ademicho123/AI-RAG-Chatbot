import requests
import streamlit as st
import logging
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Together AI API key and URL from .env
api_key = os.getenv("TOGETHER_AI_API_KEY")
api_url = os.getenv("TOGETHER_AI_API_URL")

if not api_key or not api_url:
    logger.error("TOGETHER_AI_API_KEY or TOGETHER_AI_API_URL not found in .env file")
    raise ValueError("TOGETHER_AI_API_KEY or TOGETHER_AI_API_URL not found in .env file")

def generate_section(prompt, max_retries=3):
    """
    Generate a resume section using Together AI API with retry mechanism
    
    Args:
        prompt (str): Prompt for generating resume section
        max_retries (int): Number of retry attempts
    
    Returns:
        str: Generated resume section content
    """
    if not prompt.strip():  # Check if prompt is empty
        logger.warning("Empty prompt provided")
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional resume writer helping to create a targeted resume section. Provide concise, professional content. If you cannot generate a full section, provide a minimal professional response."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
        
        try:
            logger.info(f"API Request Attempt {attempt + 1} for prompt: {prompt[:100]}...")
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            # Log full response for debugging
            logger.info(f"API Response Status Code: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Extract generated text based on the API's response structure
                generated_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if generated_text:
                    logger.info("Successfully generated section")
                    return generated_text
            
            # Log error details if request was unsuccessful
            logger.error(f"API Error on attempt {attempt + 1}: {response.status_code}")
            logger.error(f"Response Text: {response.text}")
        
        except requests.RequestException as e:
            logger.error(f"Request Exception on attempt {attempt + 1}: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
    
    # If all retries fail, return a generic placeholder
    logger.warning("Failed to generate section after all retries")
    return f"Professional {prompt.split()[0]} section could not be generated."

def generate_resume(job_title, additional_info=""):
    """
    Generate a complete resume for a given job title
    
    Args:
        job_title (str): Target job title
        additional_info (str, optional): Additional context for resume generation
    
    Returns:
        dict: Dictionary of resume sections
    """
    if not job_title.strip():  # Check if job title is empty
        logger.warning("Empty job title provided")
        raise ValueError("Job title cannot be empty")

    sections = {}
    prompts = {
        "Summary": f"Write a 2-3 sentence professional summary for a candidate applying for a {job_title} position. Highlight key strengths, career objectives, and professional expertise.",
        "Experience": f"Create a professional experience section for a {job_title} role. Write 2-3 compelling job descriptions with quantifiable achievements and career progression.",
        "Education": f"Write a comprehensive education section for a {job_title} candidate. Include relevant degrees, academic achievements, and educational background.",
        "Skills": f"List 6-8 critical technical and professional skills specifically relevant to a successful {job_title}. Include a mix of hard and soft skills.",
        "Certifications": f"Provide 2-3 professional certifications or training that would be most valuable and relevant for a {job_title} role."
    }
    
    for section, prompt in prompts.items():
        # Append additional info to each section's prompt if available
        if additional_info.strip():
            prompt += f" Consider the following additional context: {additional_info}"
        
        try:
            section_content = generate_section(prompt)
            if section_content:
                sections[section] = section_content
        except Exception as e:
            logger.error(f"Failed to generate {section} section: {e}")
    
    if not sections:
        logger.warning("No resume sections could be generated")
        # Fallback to generating at least the summary
        summary = generate_section(f"Write a 2-3 sentence professional summary for a {job_title} position.")
        if summary:
            sections["Summary"] = summary
    
    return sections