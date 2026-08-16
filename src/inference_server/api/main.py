from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from inference_server.api.schemas import GenerateRequest
from inference_server.generation.generator import Generator
from inference_server.model.hf_model import HFModel
from inference_server.sampler.sampler import Sampler
from inference_server.tokenizer.hf_tokenizer import HFTokenizer

app = FastAPI()

tokenizer = HFTokenizer()
model = HFModel()
sampler = Sampler()
generator = Generator(tokenizer, model, sampler)

@app.get("/")
def root():
    return {"message": "Inference Server Running"}

@app.post("/generate")
def generate(request: GenerateRequest):
    return StreamingResponse(
        generator.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            penalty=request.penalty,
        ),
        media_type="text/plain; charset=utf-8",
    )