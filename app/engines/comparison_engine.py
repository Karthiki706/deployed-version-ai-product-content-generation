import re
from typing import Dict, List, Any, Tuple
from app.models.product import Product
from app.core.logger import logger
from app.engines.normalizer import NormalizationEngine

class ComparisonEngine:
    """Performs structured, deterministic field-by-field comparisons to detect AI hallucinations."""

    @staticmethod
    def compare(
        product: Product, 
        text_blob: str, 
        specs_dict: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[Dict[str, Any]], float]:
        """
        Compares normalized input fields with the generated content and specs.
        
        Returns:
            Tuple: (violations, missing_fields, unsupported_claims, comparison_report, calculated_score)
        """
        violations = []
        missing_fields = []
        unsupported_claims = []
        comparison_report = []

        # Fallback if specs_dict is a string instead of a dictionary
        if isinstance(specs_dict, str):
            parsed_specs = {}
            for line in specs_dict.split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    parsed_specs[parts[0].strip()] = parts[1].strip()
            specs_dict = parsed_specs

        norm_specs_dict = NormalizationEngine.normalize_value(specs_dict) or {}
        blob_lower = text_blob.lower()
        blob_norm = str(NormalizationEngine.normalize_value(text_blob))

        # Helper to check if a word/phrase is mentioned in the generated text
        def is_mentioned(val_norm: Any) -> bool:
            if not val_norm:
                return False
            val_str = str(val_norm).strip()
            # Check direct substring
            if val_str in blob_norm:
                return True
            # Word boundary check for short words
            if len(val_str) < 5:
                return bool(re.search(rf"\b{re.escape(val_str)}\b", blob_norm))
            return False

        # Helper to check if two normalized values match (using substring/equality check)
        def values_match(val1: Any, val2: Any) -> bool:
            if val1 == val2:
                return True
            if isinstance(val1, str) and isinstance(val2, str):
                v1 = val1.strip()
                v2 = val2.strip()
                if not v1 or not v2:
                    return False
                return v1 in v2 or v2 in v1
            return False

        # Helper to log failed comparisons
        def log_failure(attr: str, exp_raw: Any, gen_raw: Any, norm_exp: Any, norm_gen: Any, rule: str, reason: str):
            logger.warning(
                f"❌ Comparison Mismatch for '{attr}':\n"
                f"  - Expected Value: {exp_raw}\n"
                f"  - Generated Value: {gen_raw}\n"
                f"  - Normalized Expected: {norm_exp}\n"
                f"  - Normalized Generated: {norm_gen}\n"
                f"  - Comparison Rule Used: {rule}\n"
                f"  - Reason for Failure: {reason}"
            )

        # ----------------------------------------------------
        # 1. BRAND (Weight: 10)
        # ----------------------------------------------------
        brand_passed = True
        input_brand_norm = NormalizationEngine.normalize_value(product.brand)
        gen_brand_val = None

        # Try to find brand in specs_dict
        for k, v in norm_specs_dict.items():
            if "brand" in k:
                gen_brand_val = v
                break

        if gen_brand_val is None:
            # Look in text blob
            if is_mentioned(input_brand_norm):
                gen_brand_val = input_brand_norm
            else:
                # Check if competitor brand is mentioned
                competitor_brands = ["apple", "samsung", "nike", "adidas", "sony", "dell", "hp", "lg", "whirlpool"]
                found_comp = None
                for cb in competitor_brands:
                    if cb != input_brand_norm and re.search(rf"\b{cb}\b", blob_lower):
                        found_comp = cb
                        break
                if found_comp:
                    gen_brand_val = found_comp
                else:
                    gen_brand_val = "not mentioned"

        if gen_brand_val == "not mentioned":
            # Omitted but not incorrect
            comparison_report.append({
                "attribute": "Brand",
                "input": product.brand,
                "generated": "Not mentioned",
                "status": "PASS"
            })
            missing_fields.append("brand")
        elif values_match(gen_brand_val, input_brand_norm):
            comparison_report.append({
                "attribute": "Brand",
                "input": product.brand,
                "generated": str(gen_brand_val).capitalize(),
                "status": "PASS"
            })
        else:
            brand_passed = False
            issue_msg = f"Brand mismatch: AI claimed brand is '{gen_brand_val}', expected '{product.brand}'."
            violations.append({"field": "brand", "issue": issue_msg})
            unsupported_claims.append(f"Incorrect brand: '{gen_brand_val}'")
            comparison_report.append({
                "attribute": "Brand",
                "input": product.brand,
                "generated": str(gen_brand_val).capitalize(),
                "status": "FAIL"
            })
            log_failure(
                "Brand", product.brand, gen_brand_val, input_brand_norm, gen_brand_val,
                "Brand equality", "Generated brand name does not match input brand"
            )

        # ----------------------------------------------------
        # 2. MATERIAL (Weight: 20)
        # ----------------------------------------------------
        material_passed = True
        input_mat_norm = NormalizationEngine.normalize_value(product.material)
        gen_mat_val = None

        for k, v in norm_specs_dict.items():
            if "material" in k:
                gen_mat_val = v
                break

        if gen_mat_val is None:
            if is_mentioned(input_mat_norm):
                gen_mat_val = input_mat_norm
            else:
                # Check if other materials are claimed
                common_materials = ["leather", "titanium", "carbon fiber", "gorilla glass", "steel", "cotton", "plastic", "wood", "aluminum", "silicone", "gold", "silver"]
                found_m = None
                for m in common_materials:
                    if m != input_mat_norm and m in blob_lower:
                        found_m = m
                        break
                if found_m:
                    gen_mat_val = found_m
                else:
                    gen_mat_val = "not mentioned"

        if gen_mat_val == "not mentioned":
            comparison_report.append({
                "attribute": "Material",
                "input": product.material,
                "generated": "Not mentioned",
                "status": "PASS"
            })
        elif values_match(gen_mat_val, input_mat_norm):
            comparison_report.append({
                "attribute": "Material",
                "input": product.material,
                "generated": str(gen_mat_val).capitalize(),
                "status": "PASS"
            })
        else:
            material_passed = False
            issue_msg = f"Material mismatch: AI claimed material is '{gen_mat_val}', expected '{product.material}'."
            violations.append({"field": "material", "issue": issue_msg})
            unsupported_claims.append(f"Incorrect material: '{gen_mat_val}'")
            comparison_report.append({
                "attribute": "Material",
                "input": product.material,
                "generated": str(gen_mat_val).capitalize(),
                "status": "FAIL"
            })
            log_failure(
                "Material", product.material, gen_mat_val, input_mat_norm, gen_mat_val,
                "Material equality", "Generated material name does not match input material"
            )

        # ----------------------------------------------------
        # 3. WARRANTY (Weight: 10)
        # ----------------------------------------------------
        warranty_passed = True
        if not product.warranty:
            # Automatically pass since no warranty was requested
            comparison_report.append({
                "attribute": "Warranty",
                "input": "None specified",
                "generated": "Not mentioned",
                "status": "PASS"
            })
        else:
            input_warr_norm = NormalizationEngine.normalize_value(product.warranty)
            gen_warr_val = None

            for k, v in norm_specs_dict.items():
                if "warranty" in k or "guarantee" in k:
                    gen_warr_val = v
                    break

            if gen_warr_val is None:
                if is_mentioned(input_warr_norm):
                    gen_warr_val = input_warr_norm
                else:
                    # Look for durations
                    match = re.search(r"(\d+)\s*(?:year|yr|month|mth|day)", blob_lower)
                    if match:
                        gen_warr_val = NormalizationEngine.normalize_value(match.group(0))
                    else:
                        gen_warr_val = "not mentioned"

            if gen_warr_val == "not mentioned":
                comparison_report.append({
                    "attribute": "Warranty",
                    "input": product.warranty,
                    "generated": "Not mentioned",
                    "status": "PASS"
                })
            elif values_match(gen_warr_val, input_warr_norm):
                comparison_report.append({
                    "attribute": "Warranty",
                    "input": product.warranty,
                    "generated": str(gen_warr_val).capitalize(),
                    "status": "PASS"
                })
            else:
                warranty_passed = False
                issue_msg = f"Warranty mismatch: AI claimed warranty is '{gen_warr_val}', expected '{product.warranty}'."
                violations.append({"field": "warranty", "issue": issue_msg})
                unsupported_claims.append(f"Incorrect warranty claim: '{gen_warr_val}'")
                comparison_report.append({
                    "attribute": "Warranty",
                    "input": product.warranty,
                    "generated": str(gen_warr_val),
                    "status": "FAIL"
                })
                log_failure(
                    "Warranty", product.warranty, gen_warr_val, input_warr_norm, gen_warr_val,
                    "Warranty matching", "Generated warranty duration does not match input warranty specification"
                )

        # ----------------------------------------------------
        # 4. SPECIFICATIONS (Weight: 25)
        # Includes: Weight, Dimensions, and Custom Specifications
        # ----------------------------------------------------
        spec_items = []  # List of tuples: (name, input_val, gen_val, passed)

        # A. Weight Check
        if product.weight_grams is not None:
            gen_w_val = None
            
            # Find weight in specs_dict
            for k, v in norm_specs_dict.items():
                if "weight" in k or "mass" in k:
                    gen_w_val = v
                    break

            if gen_w_val is None:
                # Regex weight extraction
                match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|gm|gms|grams|kg|kilograms|kgs)\b", blob_lower)
                if match:
                    gen_w_val = NormalizationEngine.normalize_value(match.group(0))
                else:
                    gen_w_val = "not mentioned"

            # Parse float value of generated weight to compare numerically
            gen_grams = None
            if gen_w_val and gen_w_val != "not mentioned":
                try:
                    cleaned_val = str(gen_w_val).replace(" g", "").strip()
                    gen_grams = float(cleaned_val)
                except Exception:
                    pass

            if gen_w_val == "not mentioned":
                spec_items.append(("Weight", product.weight_raw, "Not mentioned", True))
            elif gen_grams is not None and abs(gen_grams - product.weight_grams) < 0.01:
                spec_items.append(("Weight", product.weight_raw, str(gen_w_val), True))
            else:
                spec_items.append(("Weight", product.weight_raw, str(gen_w_val), False))
                violations.append({
                    "field": "weight",
                    "issue": f"Weight mismatch: AI claimed weight '{gen_w_val}', expected '{product.weight_raw}'."
                })
                unsupported_claims.append(f"Incorrect weight claim: '{gen_w_val}'")
                log_failure(
                    "Weight", product.weight_raw, gen_w_val, f"{product.weight_grams} g", gen_w_val,
                    "Weight unit conversion", "Weight values do not align when normalized to grams"
                )

        # B. Dimensions Check
        for dim_key, dim_val in product.dimensions.items():
            exp_dim_norm = float(dim_val)
            gen_dim_val = None
            
            # Search specs_dict
            for k, v in norm_specs_dict.items():
                if dim_key.lower() in k:
                    gen_dim_val = v
                    break

            if gen_dim_val is None:
                # Look in text blob near dimension key
                match = re.search(rf"{dim_key.lower()}\b.*?\b(\d+(?:\.\d+)?)", blob_lower)
                if match:
                    gen_dim_val = float(match.group(1))
                else:
                    gen_dim_val = "not mentioned"

            if gen_dim_val == "not mentioned":
                spec_items.append((f"Dimension: {dim_key}", f"{dim_val}", "Not mentioned", True))
            elif isinstance(gen_dim_val, (int, float)) and abs(gen_dim_val - exp_dim_norm) < 0.01:
                spec_items.append((f"Dimension: {dim_key}", f"{dim_val}", f"{gen_dim_val}", True))
            else:
                spec_items.append((f"Dimension: {dim_key}", f"{dim_val}", f"{gen_dim_val}", False))
                violations.append({
                    "field": "dimensions",
                    "issue": f"Dimension mismatch for '{dim_key}': AI claimed '{gen_dim_val}', expected '{dim_val}'."
                })
                unsupported_claims.append(f"Dimension '{dim_key}' mismatch")
                log_failure(
                    f"Dimension: {dim_key}", dim_val, gen_dim_val, exp_dim_norm, gen_dim_val,
                    "Numeric dimension check", "Dimension floating point values do not match within tolerance (0.01)"
                )

        # C. Custom Specs Check
        for spec_key, spec_val in product.specifications.items():
            norm_key = spec_key.lower().replace("-", " ").strip()
            norm_val_exp = NormalizationEngine.normalize_value(spec_val)
            gen_spec_val = None

            # Look in normalized specs dict
            for k, v in norm_specs_dict.items():
                if norm_key in k:
                    gen_spec_val = v
                    break

            if gen_spec_val is None:
                # Look in normalized text blob
                if is_mentioned(norm_val_exp):
                    gen_spec_val = norm_val_exp
                else:
                    gen_spec_val = "not mentioned"

            if gen_spec_val == "not mentioned":
                spec_items.append((spec_key, str(spec_val), "Not mentioned", True))
            elif values_match(gen_spec_val, norm_val_exp):
                spec_items.append((spec_key, str(spec_val), str(gen_spec_val), True))
            else:
                spec_items.append((spec_key, str(spec_val), str(gen_spec_val), False))
                violations.append({
                    "field": "specifications",
                    "issue": f"Specification mismatch for '{spec_key}': AI claimed '{gen_spec_val}', expected '{spec_val}'."
                })
                unsupported_claims.append(f"Specification mismatch: {spec_key}")
                log_failure(
                    spec_key, spec_val, gen_spec_val, norm_val_exp, gen_spec_val,
                    "Custom specification matching", "Generated specification value does not match expected value"
                )

        # Add specs checks to comparison report
        passed_specs = 0
        for attr_name, exp_v, gen_v, passed in spec_items:
            if passed:
                passed_specs += 1
            comparison_report.append({
                "attribute": attr_name,
                "input": exp_v,
                "generated": gen_v,
                "status": "PASS" if passed else "FAIL"
            })

        if spec_items:
            specifications_score = (passed_specs / len(spec_items)) * 25.0
        else:
            specifications_score = 25.0  # Pass default if none present

        # ----------------------------------------------------
        # 5. FEATURES (Weight: 25)
        # ----------------------------------------------------
        passed_features = 0
        if not product.features:
            features_score = 25.0
            comparison_report.append({
                "attribute": "Features",
                "input": "None specified",
                "generated": "Not mentioned",
                "status": "PASS"
            })
        else:
            for feature in product.features:
                feat_norm = NormalizationEngine.normalize_value(feature)
                # Split feature into words and check matches
                feat_words = [w for w in re.findall(r"\w+", str(feat_norm)) if len(w) > 3]
                matched = False
                
                if is_mentioned(feat_norm):
                    matched = True
                elif feat_words:
                    # Allow fuzzy match if 70% of the long words are in blob_norm
                    matches = sum(1 for w in feat_words if w in blob_norm)
                    if (matches / len(feat_words)) >= 0.7:
                        matched = True

                if matched:
                    passed_features += 1
                    comparison_report.append({
                        "attribute": f"Feature: {feature[:30]}...",
                        "input": feature,
                        "generated": "Integrated in content",
                        "status": "PASS"
                    })
                else:
                    missing_fields.append(f"feature: {feature}")
                    comparison_report.append({
                        "attribute": f"Feature: {feature[:30]}...",
                        "input": feature,
                        "generated": "Omitted / Rephrased",
                        "status": "FAIL"
                    })
                    logger.info(f"Omitted Feature: '{feature}' was not directly integrated into the generated text.")

            features_score = (passed_features / len(product.features)) * 25.0

        # ----------------------------------------------------
        # 6. PACKAGE CONTENTS (Weight: 10)
        # ----------------------------------------------------
        passed_pkg = 0
        
        # Build comprehensive search text from text_blob and specs_dict (both keys and values)
        specs_str = " ".join([f"{k} {v}" for k, v in norm_specs_dict.items()])
        full_gen_text = f"{text_blob} {specs_str}"
        full_gen_text_norm = str(NormalizationEngine.normalize_value(full_gen_text))

        if not product.package_contents:
            pkg_score = 10.0
            comparison_report.append({
                "attribute": "Package Contents",
                "input": "None specified",
                "generated": "Not mentioned",
                "status": "PASS"
            })
        else:
            for item in product.package_contents:
                item_norm = NormalizationEngine.normalize_value(item)
                matched = False
                
                if item_norm:
                    item_norm_str = str(item_norm).strip()
                    if item_norm_str in full_gen_text_norm:
                        matched = True
                    else:
                        # Treat singular/plural differences
                        words = [w.strip() for w in re.findall(r"\b\w+\b", item_norm_str) if len(w) > 2]
                        if words:
                            def make_singular(w: str) -> str:
                                w = w.lower()
                                if w.endswith("ies") and len(w) > 3:
                                    return w[:-3] + "y"
                                if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
                                    return w[:-1]
                                return w
                                
                            singular_words = [make_singular(w) for w in words]
                            search_words = [make_singular(w) for w in re.findall(r"\b\w+\b", full_gen_text_norm) if len(w) > 2]
                            
                            matched_words = 0
                            for sw in singular_words:
                                if any(sw in w or w in sw for w in search_words):
                                    matched_words += 1
                            
                            if (matched_words / len(words)) >= 0.8:
                                matched = True

                if matched:
                    passed_pkg += 1
                    comparison_report.append({
                        "attribute": f"Package Content: {item}",
                        "input": item,
                        "generated": "Mentioned",
                        "status": "PASS"
                    })
                else:
                    violations.append({
                        "field": "package_contents",
                        "issue": f"Missing package content item: '{item}'."
                    })
                    missing_fields.append(f"package_content: {item}")
                    comparison_report.append({
                        "attribute": f"Package Content: {item}",
                        "input": item,
                        "generated": "Missing",
                        "status": "FAIL"
                    })
                    log_failure(
                        f"Package Content: {item}", item, "Missing", item_norm, "Missing",
                        "Package content matching", "Omitted mandatory packaging contents"
                    )

            pkg_score = (passed_pkg / len(product.package_contents)) * 10.0

        # ----------------------------------------------------
        # 7. PRODUCT NAME, CATEGORY, TARGET AUDIENCE (Log only, no separate weights)
        # Check title contains product name
        # ----------------------------------------------------
        name_norm = NormalizationEngine.normalize_value(product.product_name)
        if not is_mentioned(name_norm):
            logger.warning(f"Product Name '{product.product_name}' was not detected in the generated content title/text.")

        # Calculate final mathematical score
        brand_score = 10.0 if brand_passed else 0.0
        material_score = 20.0 if material_passed else 0.0
        warr_score = 10.0 if warranty_passed else 0.0
        
        calculated_score = (
            brand_score + 
            material_score + 
            warr_score + 
            specifications_score + 
            features_score + 
            pkg_score
        )
        calculated_score = max(0.0, min(100.0, float(calculated_score)))

        return violations, missing_fields, unsupported_claims, comparison_report, calculated_score
