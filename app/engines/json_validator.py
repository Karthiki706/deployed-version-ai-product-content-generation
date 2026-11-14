import json
from pydantic import ValidationError
from app.models.response import GeneratedContent
from app.core.logger import logger

class JSONValidationEngine:
    """Validates LLM output against the GeneratedContent Pydantic model."""
    
    @staticmethod
    def validate(raw_content: str) -> GeneratedContent:
        logger.info("Validating Gemini response text against Pydantic schema.")
        
        # Parse JSON format
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error(f"Raw string is not valid JSON: {str(e)}")
            raise ValueError(f"JSON format error: {str(e)}")
            
        # Validate schema using Pydantic model
        try:
            validated_content = GeneratedContent.model_validate(data)
            logger.info("Response successfully validated against Pydantic schema.")
            return validated_content
        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise ValueError(f"JSON schema mismatch: {str(e)}")
