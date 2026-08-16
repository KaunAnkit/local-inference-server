from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int | None = 50
    top_p: float | None = 0.9
    penalty: float = 1.0