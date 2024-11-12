# AI Resume Generator

This project leverages Hugging Face's Flan-T5 model to automatically generate professional resumes based on a given job title and additional information. The app is built using Streamlit and integrates various Python libraries to provide an intuitive and interactive user experience, allowing users to generate and download personalized resumes in PDF format.

## Features

- **Job Title Input**: Allows users to specify the job title (e.g., Senior Software Engineer).
- **Additional Information**: Provides a text area to input extra details that may help in resume generation (e.g., required skills, experience, etc.).
- **Generated Resume**: AI generates a professional resume with sections such as Summary, Experience, Education, Skills, and Certifications.
- **Download PDF**: Users can download the generated resume as a PDF file.

## Installation

### Requirements
- Python 3.8+
- Streamlit
- Hugging Face Transformers library
- ReportLab (for PDF generation)
- PyPDF (for handling PDF files)
- dotenv (for managing environment variables)

### Steps to Install
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ai-resume-generator.git
    cd ai-resume-generator
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up your Hugging Face token:
    - Create a `.env` file in the root directory and add the following line:
      ```
      HUGGINGFACE_TOKEN=your_huggingface_token
      ```

## Running the App

To start the Streamlit app, run the following command:
```bash
streamlit run app.py
This tool creates tailored, high-quality resumes instantly.
