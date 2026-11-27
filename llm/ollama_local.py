import httpx
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ValidationError

from config.settings import OLLAMA_CHAT_MODEL, OLLAMA_EMBEDDING_MODEL, OLLAMA_URL


class OllamaError(RuntimeError):
    """The local model service did not return a usable response."""


def _request(client: httpx.Client, endpoint: str, payload: dict) -> dict:
    try:
        response = client.post(endpoint, json=payload)
    except httpx.TimeoutException as exc:
        raise OllamaError("Ollama took too long to respond.") from exc
    except httpx.RequestError as exc:
        raise OllamaError("Cannot reach Ollama on 127.0.0.1:11434. Run ollama serve.") from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise OllamaError("Ollama did not return a JSON response.") from exc
    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an unexpected response.")
    if response.is_error:
        model = payload["model"]
        if response.status_code == 404:
            raise OllamaError(f"Ollama model {model!r} is unavailable. Run ollama pull {model}.")
        raise OllamaError(f"Ollama returned an error: {result.get('error', response.reason_phrase)}")
    return result


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = OLLAMA_EMBEDDING_MODEL):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        batch_size = 32
        with httpx.Client(base_url=OLLAMA_URL, timeout=180, trust_env=False) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                result = _request(client, "/api/embed", {
                    "model": self.model,
                    "input": batch,
                    "truncate": True,
                })
                batch_vectors = result.get("embeddings", [])
                if len(batch_vectors) != len(batch):
                    raise OllamaError("Ollama did not return an embedding for each passage.")
                vectors.extend(batch_vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _chat(messages: list[dict[str, str]], schema: dict | None = None) -> str:
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 900},
    }
    if schema is not None:
        payload["format"] = schema

    with httpx.Client(base_url=OLLAMA_URL, timeout=300, trust_env=False) as client:
        result = _request(client, "/api/chat", payload)

    content = result.get("message", {}).get("content", "").strip()
    if not content:
        raise OllamaError("Ollama returned an empty response.")
    return content


def get_chat_completion(system_prompt: str, prompt: str) -> str:
    return _chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ])


def get_structured_completion(prompt: str, response_model: type[BaseModel]) -> BaseModel:
    result = _chat(
        [{"role": "user", "content": prompt}],
        schema=response_model.model_json_schema(),
    )
    try:
        return response_model.model_validate_json(result)
    except ValidationError as exc:
        raise OllamaError("Ollama did not return valid financial metrics.") from exc
