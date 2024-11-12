from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict
from huggingface_hub import login
from dotenv import load_dotenv
import os

class ResumeGenerator:
    def __init__(self, model_name: str = "google/flan-t5-xl"):
        """
        Initialize the Resume Generator with a specified model.
        Args:
            model_name (str): Name of the model to use from HuggingFace
        """
        # Load environment variables
        load_dotenv()
        
        # Get token from .env
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
            
        # Login to Hugging Face
        login(token=hf_token)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
    def generate_resume(self, job_title: str, additional_info: str = "") -> Dict[str, str]:
        """
        Generate a resume for a given job title.
    
        Args:
            job_title (str): The target job title
            additional_info (str): Any additional information to consider
        
        Returns:
            Dict[str, str]: Dictionary containing resume sections
        """
        # Craft prompts for each section based on job title
        sections = {}
    
        # Generate Summary
        summary_prompt = f"""Task: Write a 2-3 sentence professional summary for a {job_title} resume.
        The summary should highlight key qualifications, relevant expertise, and fit for the {job_title} role.
        Additional context: {additional_info}
        Output only the summary section, without any other headers or sections."""
    
        sections["Summary"] = self._generate_section(summary_prompt)
    
        # Generate Experience
        experience_prompt = f"""Task: Write a professional experience section for a {job_title} resume.
        Include 2-3 most recent and relevant positions, highlighting responsibilities, achievements, and skills developed in each role.
        Additional context: {additional_info}
        Output only the experience section, without any other headers or sections."""
    
        sections["Experience"] = self._generate_section(experience_prompt)
    
        # Generate Education
        education_prompt = f"""Task: Write an education section for a {job_title} resume.
        List the highest degree(s) and any relevant certifications or coursework.
        Additional context: {additional_info} 
        Output only the education section, without any other headers or sections."""
    
        sections["Education"] = self._generate_section(education_prompt)
    
        # Generate Skills
        skills_prompt = f"""Task: List the top 5-7 technical skills and 3-5 professional skills relevant for a {job_title} position.
        Technical skills should include proficiencies in areas pertinent to a {job_title} role.
        Professional skills should cover abilities like problem-solving, communication, collaboration, etc.
        Additional context: {additional_info}
        Output the skills in a bullet-pointed list, without any other headers or sections."""
    
        sections["Skills"] = self._generate_section(skills_prompt)
    
        # Generate Certifications
        cert_prompt = f"""Task: List any relevant professional certifications for a {job_title} role.
        Additional context: {additional_info}
        Output the certifications in a bullet-pointed list, without any other headers or sections."""
    
        sections["Certifications"] = self._generate_section(cert_prompt)
    
        return {k: v.strip() for k, v in sections.items() if v.strip()}

    
    def _generate_section(self, prompt: str) -> str:
        """
        Generate a single section of the resume.
        
        Args:
            prompt (str): The prompt for the section
            
        Returns:
            str: Generated text for the section
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=300,
            min_length=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)