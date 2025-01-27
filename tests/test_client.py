import os
import pytest
from unittest.mock import patch
from openai_api_client.client import OpenAIClient


@pytest.fixture
def mock_openai_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "Machine Learning is a field of artificial intelligence..."
                }
            }
        ]
    }


@patch("openai_api_client.client.OpenAIClient.get_chat_completion")
def test_get_chat_completion(mock_get_chat_completion, mock_openai_response):
    mock_get_chat_completion.return_value = mock_openai_response

    api_key = os.getenv("OPENAI_API_KEY", "test_key")
    openai_client = OpenAIClient(api_key=api_key)
    chat_completion = openai_client.get_chat_completion(
        "gpt-4o-mini",
        [
            {
                "role": "user",
                "content": "What is Machine Learning?",
            }
        ],
    )
    assert chat_completion["choices"][0]["message"]["content"].startswith(
        "Machine Learning"
    )
    assert len(chat_completion["choices"]) == 1
