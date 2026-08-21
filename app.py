import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to parse PDF: {str(e)}")

def build_prompt(resume_text: str, job_description: str) -> str:
    """Builds the prompt for the LLM analysis."""
    prompt = f"""
You are an expert technical recruiter and resume analyzer. Your task is to evaluate a candidate's resume against a job description.

**Core Requirements for Analysis:**
1. Explicitly extract and identify the following sections from the resume (if present): **skills, experience, education, certifications, projects, and languages**.
2. Use **hybrid matching**: weight exact keyword matches higher, but also explicitly recognize and surface semantic skill equivalents (e.g., "Python" matching "scripting", "Agile" matching "Scrum", or "project management" matching "cross-functional coordination").
3. Provide a **Fit Score**: A numerical score from 0-100% indicating how well the resume aligns with the job description based on the hybrid matching.
4. Highlight **Strengths**: Specific sections and skills from the resume that directly match (exact or semantic) the job requirements.
5. Identify **Improvements**: Concrete, actionable suggestions to better align the resume with the job description.

---
**Resume text:**
{resume_text}

---
**Job Description:**
{job_description}

---
Format your response as a professional Markdown report, including clear headings for "Fit Score", "Extracted Resume Sections", "Strengths", and "Improvements". Do not include any JSON or other formatting, just the markdown report.
"""
    return prompt

def analyze_resume(prompt: str, api_key: str, model: str) -> str:
    """Calls OpenAI API to generate the analysis."""
    try:
        client = OpenAI(api_key=api_key)
        logger.info(f"Sending request to OpenAI using model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful expert technical recruiter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise RuntimeError(f"Failed to communicate with OpenAI: {str(e)}")

def main():
    st.set_page_config(page_title="AI Resume & Job Matcher", page_icon="📄", layout="wide")
    
    st.title("📄 AI Resume & Job Matcher")
    st.markdown("Upload your resume and a job description to get a detailed fit analysis using OpenAI.")
    
    # Sidebar
    st.sidebar.header("Settings & Uploads")
    
    # API Key Input
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", help="Your API key is only used for this session and is not saved.")
    
    selected_model = st.sidebar.selectbox("Select Model", options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])

    st.sidebar.markdown("---")
    
    # File uploads
    st.sidebar.subheader("Upload Documents")
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF only)", type=["pdf"])
    
    jd_input_type = st.sidebar.radio("Job Description Input Method", ["Text Paste", "PDF Upload"])
    
    jd_text = ""
    if jd_input_type == "Text Paste":
        jd_text = st.sidebar.text_area("Paste Job Description here:", height=200)
    else:
        jd_file = st.sidebar.file_uploader("Upload Job Description (PDF)", type=["pdf"])
        if jd_file:
            try:
                jd_text = extract_text_from_pdf(jd_file)
                st.sidebar.success("Job description parsed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error parsing Job Description PDF: {e}")
                
    analyze_button = st.sidebar.button("Analyze Fit", type="primary")
    
    # Main content area
    if analyze_button:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
        if not resume_file:
            st.error("Please upload a resume.")
            return
        if not jd_text.strip():
            st.error("Please provide a job description.")
            return
            
        with st.spinner("Parsing resume..."):
            try:
                resume_text = extract_text_from_pdf(resume_file)
            except Exception as e:
                st.error(f"Failed to read resume: {e}")
                return
                
        if not resume_text.strip():
            st.error("Resume PDF is empty or could not be parsed.")
            return
            
        prompt = build_prompt(resume_text, jd_text)
        
        with st.spinner(f"Analyzing match using '{selected_model}'... This may take a minute."):
            try:
                analysis_result = analyze_resume(prompt, api_key, selected_model)
                st.success("Analysis complete!")
                
                st.markdown("### Analysis Report")
                st.markdown(analysis_result)
                
                # Download button
                st.download_button(
                    label="Download Report as Markdown",
                    data=analysis_result,
                    file_name="resume_analysis_report.md",
                    mime="text/markdown"
                )
            except RuntimeError as e:
                st.error(f"API Error: {e}. Please check your API key and try again.")
                logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()