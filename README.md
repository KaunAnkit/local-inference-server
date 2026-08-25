# Local LLM Inference Server

A minimal production-style LLM inference server built from scratch to understand how modern language model inference works.

Instead of relying entirely on high-level libraries, this project implements the core inference pipeline step by step, including decoding algorithms, KV Cache, and a REST API for text generation.

---

## Features

- Hugging Face model integration (SmolLM2-135M)
- Custom tokenizer wrapper
- Autoregressive text generation
- KV Cache support
- Temperature sampling
- Top-k sampling
- Top-p (Nucleus) sampling
- Repetition penalty
- Streaming token generation
- FastAPI REST API
- Unit tests

---

## Project Structure

```text
src/
└── inference_server/
    ├── api/
    │   ├── main.py
    │   └── schemas.py
    │
    ├── generation/
    │   └── generator.py
    │
    ├── model/
    │   ├── hf_model.py
    │   └── model.py
    │
    ├── sampler/
    │   └── sampler.py
    │
    ├── tokenizer/
    │   ├── hf_tokenizer.py
    │   └── tokenizer.py
    │
    └── utils/
```

---

# Inference Pipeline

```
                Prompt
                   │
                   ▼
             HF Tokenizer
                   │
                   ▼
             Token IDs
                   │
                   ▼
            Language Model
         (KV Cache Enabled)
                   │
                   ▼
               Logits
                   │
                   ▼
       Repetition Penalty
                   │
                   ▼
          Temperature Scaling
                   │
                   ▼
             Top-k Sampling
                   │
                   ▼
          Top-p Sampling
                   │
                   ▼
          Sample Next Token
                   │
                   ▼
      Append Token + Update Cache
                   │
                   ▼
        Stop on EOS or Max Tokens
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/KaunAnkit/local-inference-server

cd local-inference-server
```

Install dependencies

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

# Running the API

```bash
uvicorn --app-dir src inference_server.api.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

to access the Swagger UI.

---

# API

## POST /generate

Example request

```json
{
    "prompt":"Explain what an LLM is",
    "max_new_tokens":100,
    "temperature":0.7,
    "top_k":50,
    "top_p":0.9,
    "penalty":1.2
}
```

Example response

```text
Large Language Models (LLMs) are neural networks trained on massive text datasets...
```

---

# Implemented Sampling Algorithms

### Temperature

Controls randomness during sampling.

---

### Top-k Sampling

Keeps only the k highest probability tokens before sampling.

---

### Top-p (Nucleus) Sampling

Samples from the smallest set of tokens whose cumulative probability exceeds p.

---

### Repetition Penalty

Reduces the probability of previously generated tokens to decrease repetition.

---

# KV Cache

The first forward pass processes the entire prompt.

Subsequent generations only process the latest generated token while reusing cached attention states.

This significantly reduces inference latency.

---

# Performance

| Version | Tokens/sec |
|---------|-----------:|
| Initial Implementation | ~2.1 |
| After KV Cache | ~8.8 |
| Streaming Generation | ~7.3 |

*(Benchmarks measured on my local machine using SmolLM2-135M.)*

---

# Running Tests

```bash
pytest
```

---

# Roadmap

## Completed

- [x] Tokenizer
- [x] Hugging Face model wrapper
- [x] Generation loop
- [x] KV Cache
- [x] Temperature sampling
- [x] Top-k sampling
- [x] Top-p sampling
- [x] Repetition penalty
- [x] Streaming generation
- [x] FastAPI REST API
- [x] Unit tests

## Next

- [ ] Continuous batching
- [ ] Prefix caching
- [ ] Speculative decoding
- [ ] Quantization
- [ ] Flash Attention
- [ ] OpenAI-compatible API
- [ ] Multi-request scheduling
- [ ] Benchmark suite

---

# Learning Goals

This project is built for educational purposes to better understand:

- Transformer inference
- Autoregressive decoding
- Efficient generation
- Sampling algorithms
- KV Cache
- Production inference systems

---

# Contributing

Contributions, suggestions, and discussions are welcome.

Feel free to open an issue or submit a pull request.

---

# License

MIT License