from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm.ollama_local import OllamaEmbeddings, OllamaError, get_chat_completion
from rag.context import build_context
from vectorstore.sqlite_store import LocalVectorStore

router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    company: str | None = None
    year: int | None = None


@router.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question.")

    store = LocalVectorStore(OllamaEmbeddings())
    try:
        documents = store.invoke(question, company=request.company, year=request.year)
        if not documents:
            return {"answer": "No indexed passages found. Upload a matching financial report first."}

        answer = get_chat_completion(
            system_prompt=(
                "Answer the user's question in the same language. Use only the report extracts. "
                "Give the currency, units and fiscal year when known. Cite the source file name. "
                "If the passages do not contain the answer, say so."
            ),
            prompt=f"Report extracts:\n{build_context(documents)}\n\nQuestion: {question}",
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"answer": answer}
