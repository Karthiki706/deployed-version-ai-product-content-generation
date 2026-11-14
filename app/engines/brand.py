import os
import json
import re
from typing import Dict, Any, List
from app.models.reports import BrandReport
from app.core.logger import logger

class BrandComplianceEngine:
    """Verifies that generated content conforms to brand guidelines, restricted lists, and tone constraints."""

    def __init__(self):
        self.restricted_words = self._load_restricted_words()

    def _load_restricted_words(self) -> List[str]:
        """Loads restricted words from external brand_rules.json config file, falling back to defaults."""
        default_words = ["best", "cheapest", "guaranteed", "perfect", "revolutionary", "risk-free", "ultimate", "no risk"]
        config_path = os.path.join("app", "core", "brand_rules.json")
        
        if not os.path.exists(config_path):
            logger.info(f"Brand rules config not found at {config_path}. Using default list.")
            return default_words
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                words = data.get("restricted_words", default_words)
                logger.info(f"Loaded brand compliance rules configuration successfully. Restricted words: {words}")
                return words
        except Exception as e:
            logger.error(f"Failed to load brand rules from JSON: {str(e)}. Falling back to defaults.")
            return default_words

    def check(self, content: Dict[str, Any], requested_voice: str, requested_tone: str) -> Dict[str, Any]:
        logger.info(f"Running Brand Compliance Check (Voice: {requested_voice}, Tone: {requested_tone})")
        
        violations = []
        tone_check = "PASS"
        voice_compliance = True
        writing_style_check = "PASS"
        
        text_fields = [
            content.get("title", ""),
            content.get("short_description", ""),
            content.get("long_description", ""),
            " ".join(content.get("bullet_points", []))
        ]
        full_text = " ".join(text_fields)
        full_text_lower = full_text.lower()

        # 1. Rule-Based Restricted Words Validation
        for word in self.restricted_words:
            if self._is_marketing_exaggeration(word, full_text_lower):
                violations.append(word)

        # 2. Tone Compliance Check
        exclamation_count = full_text.count("!")
        informal_slang = ["cool", "awesome", "mind-blowing", "crazy", "epic", "huge", "insane", "cheap"]
        slang_found = [s for s in informal_slang if self._is_actual_slang(s, full_text_lower)]

        is_formal_tone = requested_tone.lower() in ["expert", "professional", "informative", "authoritative"]
        
        if is_formal_tone:
            if exclamation_count > 1:
                tone_check = "WARNING: Excess exclamations for formal tone."
            if slang_found:
                tone_check = f"FAIL: Slang terms {slang_found} used in formal tone."
                writing_style_check = "FAIL"

        # 3. Brand Voice Compliance Check
        if requested_voice.lower() == "minimalist":
            long_desc = content.get("long_description", "")
            words_count = len(re.findall(r"\b\w+\b", long_desc))
            if words_count > 250:
                voice_compliance = False
                writing_style_check = "FAIL: Minimalist voice requires shorter, punchier text."

        # 4. Brand Score Calculation
        brand_score = 100.0
        
        # Deduct 15 points per restricted word violation
        brand_score -= len(violations) * 15.0
        
        # Deduct 10 points for tone issues
        if "FAIL" in tone_check:
            brand_score -= 15.0
        elif "WARNING" in tone_check:
            brand_score -= 5.0
            
        # Deduct 10 points for voice compliance issues
        if not voice_compliance:
            brand_score -= 10.0
            
        # Deduct 10 points if style check fails
        if writing_style_check == "FAIL":
            brand_score -= 10.0

        brand_score = max(0.0, brand_score)

        report = BrandReport(
            violations=violations,
            tone_check=tone_check,
            voice_compliance=voice_compliance,
            writing_style_check=writing_style_check,
            score=brand_score
        )

        logger.info(f"Brand check complete. Score: {report.score}, Violations: {report.violations}")
        return report.model_dump()

    def _is_marketing_exaggeration(self, word: str, text: str) -> bool:
        """
        Differentiates marketing exaggeration (puffery) from standard descriptive language.
        
        Returns:
            bool: True if it is a puffery violation, False if normal descriptive context.
        """
        matches = list(re.finditer(rf"\b{re.escape(word)}\b", text))
        if not matches:
            return False
            
        for m in matches:
            # Extract local context window around the match
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            context = text[start:end]
            
            if word == "best":
                # Exclude safe descriptive contexts
                safe_phrases = ["best practices", "best suited", "best seller", "best-seller", "best efforts", "best wishes", "best selling"]
                if any(phrase in context for phrase in safe_phrases):
                    continue
                return True
                
            elif word == "perfect":
                safe_phrases = ["perfect fit", "perfectly suited", "perfectly aligned", "perfect match", "perfectly matched"]
                if any(phrase in context for phrase in safe_phrases):
                    continue
                return True
                
            elif word == "cheapest":
                # Exclude simple budget material or alternative cost statements
                safe_phrases = ["cheapest material", "cheapest option"]
                if any(phrase in context for phrase in safe_phrases):
                    # Check if they also claim "price" or "cost" guarantees
                    if any(p in context for p in ["price", "cost", "guaranteed"]):
                        return True
                    continue
                return True
                
            elif word == "guaranteed":
                safe_phrases = ["warranty guaranteed", "satisfaction guaranteed", "manufacturer guaranteed"]
                if any(phrase in context for phrase in safe_phrases):
                    continue
                return True
                
            # Default: any unhandled restricted word is a violation
            return True
            
        return False

    def _is_actual_slang(self, word: str, text: str) -> bool:
        """
        Verifies if a potential slang word is actually slang in context.
        For example, 'cool' in 'keeping your feet cool and dry' is literal temperature, not slang.
        """
        matches = list(re.finditer(rf"\b{re.escape(word)}\b", text))
        if not matches:
            return False
            
        for m in matches:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 30)
            context = text[start:end]
            
            if word == "cool":
                # Check for literal temperature contexts
                literal_phrases = [
                    "cool and dry", "cool comfort", "keep cool", "keeping cool", 
                    "keeps cool", "runs cool", "run cool", "cool air", "cool temperature"
                ]
                if any(phrase in context for phrase in literal_phrases):
                    continue
                # Also check surrounding words (e.g. feet cool, device cool)
                if re.search(r"cool\s+and\s+(?:dry|comfortable|quiet)", context):
                    continue
                return True
                
            elif word == "cheap":
                # 'cheap' is slang in 'cheap product' but descriptive in 'cheap option' or cost-efficiency analysis.
                return True
                
            return True
            
        return False
