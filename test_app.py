import pytest
from unittest.mock import patch, MagicMock
from app import extract_text_from_pdf, build_prompt, analyze_resume, get_available_models
import io

def test_build_prompt_hybrid_matching_instruction():
    """Test that the prompt includes the explicit instructions for sections and hybrid matching."""
    prompt = build_prompt("Sample Resume Data", "Sample Job Data")
    
    assert "hybrid matching" in prompt.lower()
    assert "semantic skill equivalents" in prompt.lower()
    assert "Sample Resume Data" in prompt
    assert "Sample Job Data" in prompt
    assert "skills, experience, education, certifications, projects, and languages" in prompt.lower()

@patch("app.fitz.open")
def test_extract_text_from_pdf_success(mock_fitz_open):
    """Test successful PDF extraction via PyMuPDF."""
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Test PDF Content"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    fake_file = io.BytesIO(b"fake pdf content")
    text = extract_text_from_pdf(fake_file)
    
    assert text == "Test PDF Content"
    mock_fitz_open.assert_called_once()

@patch("app.fitz.open")
def test_extract_text_from_pdf_failure(mock_fitz_open):
    """Test PDF extraction error handling."""
    mock_fitz_open.side_effect = Exception("Corrupt PDF")
    
    fake_file = io.BytesIO(b"fake pdf content")
    with pytest.raises(ValueError, match="Failed to parse PDF: Corrupt PDF"):
        extract_text_from_pdf(fake_file)

@patch("app.ollama.list")
def test_get_available_models_success(mock_ollama_list):
    """Test fetching models when Ollama is running."""
    mock_ollama_list.return_value = {'models': [{'name': 'llama3:latest'}, {'name': 'mistral'}]}
    models = get_available_models()
    assert models == ['llama3:latest', 'mistral']

@patch("app.ollama.list")
def test_get_available_models_failure(mock_ollama_list):
    """Test handling Ollama connection failure gracefully."""
    mock_ollama_list.side_effect = Exception("Connection refused")
    models = get_available_models()
    assert models == []

@patch("app.ollama.generate")
def test_analyze_resume_success(mock_ollama_generate):
    """Test successful response from Ollama API."""
    mock_ollama_generate.return_value = {'response': 'Analysis Result Markdown'}
    result = analyze_resume("Test prompt", "llama3")
    assert result == "Analysis Result Markdown"
    mock_ollama_generate.assert_called_once_with(model="llama3", prompt="Test prompt")

@patch("app.ollama.generate")
def test_analyze_resume_failure(mock_ollama_generate):
    """Test handling error from Ollama API generate call."""
    mock_ollama_generate.side_effect = Exception("Model not found")
    with pytest.raises(RuntimeError, match="Failed to communicate with Ollama: Model not found"):
        analyze_resume("Test prompt", "llama3")