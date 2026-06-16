import pytest
from unittest.mock import MagicMock, patch
from classifier import classify_episode

@pytest.fixture
def mock_groq_client():
    with patch("classifier.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client

def test_classify_episode_garbage_handling(mock_groq_client):
    """Verify that a nonsensical description returns 'unknown' label."""
    # Simulate LLM returning something that doesn't follow the Label: format or is invalid
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="This is just some random text that doesn't contain a label."))]
    mock_groq_client.chat.completions.create.return_value = mock_response

    result = classify_episode("Title", "Nonsensical description", [])
    assert result["label"] == "unknown"

def test_classify_episode_normalization(mock_groq_client):
    """Verify that 'Interview' (capitalized) is parsed as 'interview'."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Label: Interview\nReasoning: It is an interview."))]
    mock_groq_client.chat.completions.create.return_value = mock_response

    result = classify_episode("Title", "Description", [])
    assert result["label"] == "interview"

def test_classify_episode_punctuation_resilience(mock_groq_client):
    """Verify that 'interview!' and other punctuated labels are parsed correctly."""
    cases = [
        ("Label: interview!", "interview"),
        ("Label: solo.", "solo"),
        ("Label: 'panel'", "panel"),
        ("Label: \"narrative\"", "narrative"),
    ]

    for response_content, expected_label in cases:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=f"{response_content}\nReasoning: ..."))]
        mock_groq_client.chat.completions.create.return_value = mock_response

        result = classify_episode("Title", "Description", [])
        assert result["label"] == expected_label, f"Failed for {response_content}"

def test_classify_episode_invalid_label(mock_groq_client):
    """Verify that an invalid label returns 'unknown'."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Label: cooking\nReasoning: Not a podcast type we know."))]
    mock_groq_client.chat.completions.create.return_value = mock_response

    result = classify_episode("Title", "Description", [])
    assert result["label"] == "unknown"
