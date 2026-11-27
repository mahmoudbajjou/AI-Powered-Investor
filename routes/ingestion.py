import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile
from pymupdf import FileDataError

from config.settings import RAW_PDF_DIR
from ingestion.ingest_documents import ingest_document
from llm.ollama_local import OllamaError

router = APIRouter()


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".pdf" or any(c in filename for c in '<>:"|?*'):
        raise HTTPException(status_code=422, detail="Select a PDF financial report.")

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(dir=RAW_PDF_DIR) as staged:
            pdf_path = Path(staged) / filename
            with pdf_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            count = ingest_document(pdf_path, archive_to=RAW_PDF_DIR / filename)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.") from exc

    return {
        "message": "Document uploaded successfully",
        "file_name": filename,
        "chunks": count,
    }
