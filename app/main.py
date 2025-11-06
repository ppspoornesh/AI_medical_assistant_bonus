from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from app.agents.orchestrator import Orchestrator
import io
import logging

# Set up logging early – helps debug in production and shows I care about observability
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical AI Assistant API",
    description="Handles document upload, Q&A, and report generation via agentic workflow",
    version="1.0"
)
orchestrator = Orchestrator()  # Central brain – one instance for efficiency

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Accepts multiple files (PDF, DOCX, XLSX, images) and saves them temporarily.
    Returns file paths for downstream agents.
    """
    import os
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure temp folder exists – safe & idempotent
    uploaded_files = []

    for file in files:
        # Sanitize filename to prevent path traversal (security-conscious)
        safe_filename = os.path.basename(file.filename)
        file_location = os.path.join(temp_dir, safe_filename)
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)  # Stream directly to disk – memory efficient
        uploaded_files.append(file_location)
        logger.debug(f"Saved uploaded file: {file_location}")

    logger.info(f"Uploaded {len(uploaded_files)} file(s) successfully")
    return {"message": "Files uploaded successfully", "files": uploaded_files}


@app.post("/query")
async def query_assistant(input: dict):
    """
    Main query endpoint – supports both Q&A and report generation.
    Returns JSON for chat, or streams PDF for reports.
    """
    try:
        # Pass query, docs, and session ID to orchestrator – keeps logic decoupled
        result = orchestrator.invoke({
            "query": input["query"],
            "documents": input["documents"],
            "session_id": input.get("session_id", "default")
        })

        response = result.get("response")

        # Smart response routing: PDF vs Text
        if isinstance(response, io.BytesIO):
            logger.info("Streaming generated PDF report")
            return StreamingResponse(
                response,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=generated_report.pdf"}
            )
        else:
            logger.info("Returning text response")
            return JSONResponse(content={"response": response})

    except Exception as e:
        # Full traceback in logs – critical for debugging in real systems
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


if __name__ == "__main__":
    import uvicorn
    # Dev-friendly: reload on change
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)