# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import query  # import your query.py module

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YRIS Sunglasses AI Assistant",
    description="AI-powered sunglasses recommendation webhook",
    version="1.0.0"
)

# Add CORS middleware if needed for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services when the app starts"""
    try:
        query.initialize_services()
        logger.info("YRIS AI Assistant services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # You might want to exit here if initialization is critical

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "YRIS AI Assistant"}

@app.post("/webhook")
async def webhook(request: Request):
    """Main webhook endpoint for processing user queries"""
    try:
        # Get request body
        body = await request.json()
        user_query = body.get("query")
        neighbor_count = body.get("neighbor_count", 5)

        # Validate input
        if not user_query:
            raise HTTPException(
                status_code=400,
                detail="Missing 'query' field in request body"
            )

        if not isinstance(user_query, str) or not user_query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query must be a non-empty string"
            )

        # Validate neighbor_count if provided
        if not isinstance(neighbor_count, int) or neighbor_count < 1:
            neighbor_count = 5

        logger.info(f"Processing query: '{user_query}' with neighbor_count: {neighbor_count}")

        # Process the query using the improved query module
        response = query.process_query(user_query, neighbor_count)
        
        # Log successful processing
        logger.info(f"Successfully processed query: '{user_query}'")
        
        return JSONResponse(content=response)

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing webhook: {str(e)}")
        return JSONResponse(
            content={
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "action": "error",
                "error": "Internal server error"
            },
            status_code=500
        )

@app.post("/query")  
async def query_endpoint(request: Request):
    """Alternative query endpoint with same functionality as webhook"""
    return await webhook(request)

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "YRIS Sunglasses AI Assistant",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/webhook",
            "query": "/query", 
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # Run locally for testing
    logger.info("Starting YRIS AI Assistant server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )