import pytest
from app.models.product import Product
from app.engines.normalizer import NormalizationEngine
from app.engines.validator import AttributeValidationEngine
from app.engines.classifier import CategoryClassificationEngine
from app.engines.comparison_engine import ComparisonEngine
from app.engines.seo import SEOValidationEngine
from app.engines.brand import BrandComplianceEngine
from app.core.circuit_breaker import CircuitBreaker

def test_normalization_engine():
    # Setup raw mock product
    raw_prod = Product(
        product_id="PROD-TEST",
        product_name="Ultra Running Shoe v2",
        brand="Speedy",
        material="synthetic mesh",
        color="blk",  # Test abbreviation color normalization
        weight_raw="320gm",  # Test raw weight string parsing
        features=["Breathable mesh", "breathable mesh", "Vibram sole"],  # Test duplicates
        target_audience="Runners",
        seo_keywords=["running shoe", "running shoe", "outdoor gear"]  # Test duplicates
    )
    
    normalized = NormalizationEngine.normalize(raw_prod)
    
    assert normalized.color == "Black"
    assert normalized.weight_grams == 320.0
    assert normalized.weight_raw == "320.0 g"
    assert len(normalized.features) == 2  # Deduplicated features
    assert len(normalized.seo_keywords) == 2  # Deduplicated & lowercased SEO keywords
    assert "running shoe" in normalized.seo_keywords

    # Test plain number weight parsing
    raw_prod_plain_weight = Product(
        product_name="Ultra Running Shoe v2",
        brand="Speedy",
        material="synthetic mesh",
        color="Black",
        weight_raw="120.5",
        features=["Breathable mesh"],
        target_audience="Runners",
        seo_keywords=["running shoe"]
    )
    normalized_plain = NormalizationEngine.normalize(raw_prod_plain_weight)
    assert normalized_plain.weight_grams == 120.5
    assert normalized_plain.weight_raw == "120.5 g"

def test_validation_engine():
    # Valid product setup
    valid_prod = Product(
        product_name="Carbon Fiber Pole",
        brand="Alpine",
        material="Carbon Fiber",
        color="Grey",
        weight_grams=150.0,
        features=["Ultralight design"],
        target_audience="Hikers",
        seo_keywords=["hiking pole"]
    )
    report = AttributeValidationEngine.validate(valid_prod)
    assert report["status"] == "PASS"
    
    # Invalid product (missing features/keywords, which is syntactically allowed by Pydantic but semantic failure in engine)
    invalid_prod = Product(
        product_name="Broken Pole",
        brand="Alpine",
        material="Carbon Fiber",
        color="Grey",
        weight_grams=150.0,
        features=[],
        target_audience="Hikers",
        seo_keywords=[]
    )
    report = AttributeValidationEngine.validate(invalid_prod)
    assert report["status"] == "FAIL"
    assert len(report["errors"]) >= 2
    assert "At least one product feature is required." in report["errors"]
    assert "At least one SEO keyword is required." in report["errors"]

    # Verify that Pydantic raises ValidationError for negative weight on instantiation
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Product(
            product_name="Negative Weight Pole",
            brand="Alpine",
            material="Carbon Fiber",
            color="Grey",
            weight_grams=-10.0,
            target_audience="Hikers"
        )


def test_category_classification_engine():
    classifier = CategoryClassificationEngine()
    
    # Test Electronics classification
    res_elec = classifier.classify("Modern Smart Laptop X1")
    assert res_elec["category"] == "Electronics"
    assert res_elec["confidence"] == 1.0
    
    # Test Footwear classification
    res_foot = classifier.classify("Vibram Outdoor Sneaker")
    assert res_foot["category"] == "Footwear"
    
    # Test fallback classification
    res_fallback = classifier.classify("Mystery Object")
    assert res_fallback["category"] == "General"
    assert res_fallback["confidence"] == 0.5

def test_comparison_engine():
    prod = Product(
        product_name="Titanium Cup",
        brand="PeakGear",
        material="Grade 2 Titanium",
        color="Grey",
        weight_grams=80.0,
        features=["Double wall insulation"],
        target_audience="Campers",
        seo_keywords=["titanium cup"],
        warranty="2 Year Warranty"
    )
    
    # 1. Test clean output (no violations)
    text_clean = "This PeakGear Titanium Cup is made of Grade 2 Titanium. The double wall insulation keeps drinks hot. Weighs 80g. Ideal for campers. 2 Year Warranty."
    violations, missing, unsupported, _, _ = ComparisonEngine.compare(prod, text_clean, "Material: Titanium")
    assert len(violations) == 0
    assert len(unsupported) == 0
    
    # 2. Test hallucinated material and weight mismatch
    text_hallucinated = "This PeakGear cup is made of leather and wood. Weighs 120 grams. Best cup ever."
    violations, missing, unsupported, _, _ = ComparisonEngine.compare(prod, text_hallucinated, "Specs: Wood")
    
    # Should flag leather and wood as unsupported materials
    assert len(unsupported) > 0
    # Should flag weight mismatch (120g instead of 80g)
    has_weight_issue = any(v["field"] == "weight" for v in violations)
    assert has_weight_issue

    # 3. Test singular/plural and formatting equivalent package contents matching
    prod_pkg = Product(
        product_name="Wireless Headset",
        brand="SoundWave",
        material="Plastic",
        color="Black",
        weight_grams=250.0,
        features=["ANC"],
        target_audience="Music lovers",
        seo_keywords=["headphones"],
        package_contents=["remote controls", "batteries", "power cable"]
    )
    # The generated text contains singular/plural equivalents: "remote control", "battery", "power cables"
    text_pkg = "The box contains a remote control, one battery, and two power cables."
    violations_pkg, missing_pkg, unsupported_pkg, _, _ = ComparisonEngine.compare(prod_pkg, text_pkg, {})
    # It should not flag any package content items as missing
    assert len(violations_pkg) == 0

def test_seo_validation_engine():
    seo_engine = SEOValidationEngine()
    
    content = {
        "title": "Ultralight Camping Stove",
        "short_description": "Perfect titanium backpacking stove.",
        "long_description": "This camping stove is the ideal ultralight camping stove for outdoor enthusiasts. Made of titanium.",
        "meta_description": "Buy the best ultralight camping stove. Designed for backpackers and outdoor campers.",
        "seo_keywords_used": ["camping stove", "ultralight camping stove"],
        "specification_summary": ""
    }
    
    # Target keywords present
    target_kws = ["camping stove", "ultralight camping stove"]
    report = seo_engine.validate(content, target_kws, 20)
    
    assert report["score"] > 50.0
    assert len(report["missing_keywords"]) == 0
    
    # Keyword missing
    target_kws_missing = ["camping stove", "notavailablekeyword"]
    report_missing = seo_engine.validate(content, target_kws_missing, 20)
    assert "notavailablekeyword" in report_missing["missing_keywords"]
    assert report_missing["score"] < report["score"]

def test_brand_compliance_engine():
    brand_engine = BrandComplianceEngine()
    
    # 1. Clean brand check
    content_clean = {
        "title": "Premium Walking Shoe",
        "short_description": "An elegant shoe.",
        "long_description": "Our shoe is constructed from high quality fabrics for comfort.",
        "bullet_points": []
    }
    report_clean = brand_engine.check(content_clean, "Professional", "Expert")
    assert report_clean["score"] == 100.0
    
    # 2. Restricted words violation check
    content_restricted = {
        "title": "The Best Shoe Ever",
        "short_description": "Cheapest options available.",
        "long_description": "This is the perfect and guaranteed shoe choice.",
        "bullet_points": []
    }
    report_res = brand_engine.check(content_restricted, "Professional", "Expert")
    # 'best', 'cheapest', 'guaranteed', 'perfect' are restricted words
    assert len(report_res["violations"]) > 0
    assert report_res["score"] < 100.0

    # 3. Literal technical descriptive slang bypass check
    content_literal_cool = {
        "title": "Breathable Active Sock",
        "short_description": "Literal cooling active sock.",
        "long_description": "Designed for keeping your feet cool and dry under intense heat.",
        "bullet_points": []
    }
    report_literal = brand_engine.check(content_literal_cool, "Professional", "Expert")
    assert "cool" not in report_literal["violations"]


def test_circuit_breaker_states():
    # Instantiate a clean circuit breaker with low threshold and cooldown
    cb = CircuitBreaker("mock-model", failure_threshold=2, recovery_time=0.5)
    
    # Manually reset state for test predictability
    cb.state = "CLOSED"
    cb.consecutive_failures = 0
    
    assert cb.can_execute() is True
    
    # Record 1 failure
    cb.record_failure()
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 1
    
    # Record 2nd failure -> Trips
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_execute() is False
    
    # Wait for recovery cooldown
    import time
    time.sleep(0.6)
    
    # Checks state transitions to HALF_OPEN
    assert cb.can_execute() is True
    assert cb.state == "HALF_OPEN"
    
    # Success -> resets to CLOSED
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.consecutive_failures == 0


def test_invalid_json_responses():
    from app.engines.json_validator import JSONValidationEngine
    
    bad_json = "This is not json {title: text}"
    with pytest.raises(ValueError) as exc:
        JSONValidationEngine.validate(bad_json)
    assert "JSON format error" in str(exc.value)


def test_custom_exceptions():
    from app.core.exceptions import GeminiTimeoutError, CircuitBreakerOpenError
    
    with pytest.raises(GeminiTimeoutError):
        raise GeminiTimeoutError("Mock Timeout")
        
    with pytest.raises(CircuitBreakerOpenError):
        raise CircuitBreakerOpenError("Mock Circuit Block")

