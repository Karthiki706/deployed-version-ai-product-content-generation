import os
import json
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()

@router.get("/{report_id}", summary="Retrieves all reports and logs linked to a Request ID")
async def get_report_by_id(report_id: str):
    """
    Finds and compiles the stored JSON outputs and logs for a specific Request ID:
    - Generated content
    - Hallucination report
    - SEO analysis
    - Brand compliance checks
    - Quality scoring matrix
    - Execution performance log
    """
    import re
    # Validate report_id format to prevent path traversal (alphanumeric, hyphens, underscores only)
    if not re.match(r"^[a-zA-Z0-9_\-]+$", report_id):
        logger.warning(f"Invalid report_id format detected: {report_id}")
        raise HTTPException(
            status_code=400,
            detail="Invalid report ID format. Access denied."
        )

    logger.info(f"Retrieving stored reports for request: {report_id}")
    
    report_files = {
        "content": f"generated_content_{report_id}.json",
        "hallucination": f"hallucination_report_{report_id}.json",
        "seo": f"seo_report_{report_id}.json",
        "brand": f"brand_report_{report_id}.json",
        "quality": f"quality_report_{report_id}.json",
        "execution_log": f"execution_log_{report_id}.json"
    }
    
    results = {}
    not_found = []
    base_dir = os.path.realpath(settings.outputs_dir)
    
    for key, filename in report_files.items():
        filepath = os.path.join(settings.outputs_dir, filename)
        real_path = os.path.realpath(filepath)
        
        # Verify the file is strictly within the outputs directory
        if not real_path.startswith(base_dir):
            logger.warning(f"Directory traversal attempt blocked: {filepath}")
            raise HTTPException(
                status_code=400,
                detail="Access denied."
            )
            
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    results[key] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading file {filepath}: {str(e)}")
                results[key] = {"error": f"Failed to read file: {str(e)}"}
        else:
            not_found.append(key)
            
    # If all reports are missing, return 404
    if len(not_found) == len(report_files):
        logger.warning(f"No reports found on disk for ID: {report_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"Report files for ID {report_id} could not be found."
        )
        
    results["missing_reports"] = not_found
    return results
