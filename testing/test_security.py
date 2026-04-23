import sys
import os

# Adds the current directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pytest
import pandas as pd
from unittest.mock import patch
from app.services.ai_logic import run_sandbox_analysis # Your entry point

def test_pii_is_redacted_before_api_call():
    # Create a dataframe with "Danger" columns and "Safe" columns
    dangerous_df = pd.DataFrame({
        "salary": [150000, 200000],
        "email": ["boss@company.com", "dev@company.com"],
        "employee_id": ["EMP001", "EMP002"],
        "performance_score": ["High", "Medium"] # Safe column
    })

    user_instruction = "Summarize the salary trends."

    # We "patch" the Gemini client so it doesn't actually call the web.
    # Note: Use the path where the client is IMPORTED, not where it's defined.
    with patch('app.services.ai_logic.gemini_client.generate_content') as mock_api:
        
        run_sandbox_analysis(dangerous_df, user_instruction)

        # Get the arguments that were passed to the mocked function
        # args[0] is usually the prompt string
        args, _ = mock_api.call_args
        sent_prompt = args[0]

        
        # 1. Check for leaking raw values
        assert "150000" not in sent_prompt, "CRITICAL: Raw salary leaked!"
        assert "boss@company.com" not in sent_prompt, "CRITICAL: Raw email leaked!"
        
        # 2. Check if the redaction logic worked (checking for your placeholder)
        assert "[REDACTED]" in sent_prompt or "salary" not in sent_prompt
        
        # 3. Check for False Positives (Make sure it didn't redact EVERYTHING)
        assert "High" in sent_prompt, "Error: Safe data was redacted by mistake."