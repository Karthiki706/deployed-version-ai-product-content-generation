import os
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, generate, reports
from app.core.config import settings
from app.core.logger import logger
from app.core.security import rate_limit, require_api_key

app = FastAPI(
    title="Enterprise Multi-Agent Product Content Generation Platform API",
    description="Production-ready FastAPI backend for generating brand-compliant, hallucination-free, and SEO-optimized product content.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "status": "Backend Running",
        "service": "Enterprise Multi-Agent Product Content Generation Platform",
        "version": "1.0.0"
    }

# Configure CORS allowed origins based on settings
origins = [x.strip() for x in settings.allowed_origins.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(health.router, prefix="/health", tags=["System Health"])
app.include_router(generate.router, prefix="", tags=["AI Content Pipeline"],
                   dependencies=[Depends(require_api_key), Depends(rate_limit)])
app.include_router(reports.router, prefix="/report", tags=["Stored Reports"],
                   dependencies=[Depends(require_api_key)])

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Content Platform API Backend...")
    logger.info(f"Target Directory for Output Reports: {os.path.abspath(settings.outputs_dir)}")
    logger.info(f"Target Directory for Prompt Templates: {os.path.abspath(settings.prompts_dir)}")
    logger.info(f"Configured Gemini Engine: {settings.gemini_model}")
    
    # Automatically initialize the shared Gemini client.
    # If credentials are missing, raise a descriptive backend error.
    try:
        from app.core.gemini_service import GeminiService
        GeminiService.initialize()
        logger.info(f"INFO: Shared Gemini client initialized successfully. Source: {settings.credential_source}")
    except Exception as e:
        err_msg = (
            f"Gemini API key not configured.\n\n"
            f"Expected one of:\n"
            f"• Environment Variable (GEMINI_API_KEY)\n"
            f"• Streamlit Secrets\n"
            f"• backend/.env\n\n"
            f"Error: {str(e)}"
        )
        logger.critical(err_msg)
        raise RuntimeError(err_msg)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Content Platform API Backend.")
