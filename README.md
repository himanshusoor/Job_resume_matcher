# AI Resume & Job Matcher

A production-ready Streamlit application that uses local LLMs (via Ollama) to analyze how well a resume matches a job description. It performs hybrid matching to detect both exact keyword matches and semantic equivalents.

## Features
- **Privacy-First**: Uses local LLMs via Ollama, so your resume and job description never leave your machine.
- **PDF Support**: Extract text directly from PDF resumes and job descriptions using PyMuPDF.
- **Hybrid Matching**: Advanced prompt engineering ensures the LLM recognizes both exact keywords and semantic skill equivalents.
- **Stateless Execution**: Ensures no data from previous analyses leaks into current runs.
- **Downloadable Reports**: Export your match analysis directly to Markdown format.

## Prerequisites

1. Install [Ollama](https://ollama.com/)
2. Run Ollama server:
   ```bash
   ollama serve