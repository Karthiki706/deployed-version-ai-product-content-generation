from typing import List
from app.core.constants import ENRICHMENT_RULES
from app.core.logger import logger

class ProductEnrichmentEngine:
    """Generates derived tags and marketing attributes based on deterministic business rules."""
    
    def enrich(self, category: str) -> List[str]:
        logger.info(f"Enriching product attributes for category: {category}")
        tags = ENRICHMENT_RULES.get(category, ["General Retail"])
        logger.info(f"Generated enrichment tags: {tags}")
        return tags
