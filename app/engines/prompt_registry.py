import os
from typing import Tuple
from app.core.config import settings
from app.core.logger import logger

class PromptRegistry:
    """Manages prompt templates as versioned files, loading them dynamically based on category."""
    
    @staticmethod
    def get_template(category: str, version: str = "v1") -> Tuple[str, str]:
        """
        Retrieves the prompt template text for a category and version.
        
        Returns:
            Tuple[str, str]: (prompt_text, resolved_version_string)
        """
        cat_clean = category.strip().lower()
        filename = f"{cat_clean}_{version}.txt"
        filepath = os.path.join(settings.prompts_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Category template '{filename}' not found at {filepath}. Falling back to default_{version}.txt")
            filename = f"default_{version}.txt"
            filepath = os.path.join(settings.prompts_dir, filename)
            
            if not os.path.exists(filepath):
                logger.error(f"Default template '{filename}' not found. Using in-memory fallback.")
                return "Focus on value, utility, durability, and key features.", f"in_memory_{version}"
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                logger.info(f"Loaded template from {filename}")
                return content, f"{cat_clean}_{version}"
        except Exception as e:
            logger.error(f"Error loading template {filename}: {str(e)}")
            return "Focus on value, utility, durability, and key features.", f"fallback_error_{version}"
