import pytest
from unittest.mock import patch, MagicMock
from app import extract_text_from_pdf, build_prompt, analyze_resume
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

@patch("app.OpenAI")
def test_analyze_resume_success(mock_openai_class):
    """Test successful response from OpenAI API."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Analysis Result Markdown"
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client
    
    result = analyze_resume("Test prompt", "fake-api-key", "gpt-4o-mini")
    assert result == "Analysis Result Markdown"
    mock_client.chat.completions.create.assert_called_once()

@patch("app.OpenAI")
def test_analyze_resume_failure(mock_openai_class):
    """Test handling error from OpenAI API call."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Key Invalid")
    mock_openai_class.return_value = mock_client
    
    with pytest.raises(RuntimeError, match="Failed to communicate with OpenAI: API Key Invalid"):
        analyze_resume("Test prompt", "fake-api-key", "gpt-4o-mini")