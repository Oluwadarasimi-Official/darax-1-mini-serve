FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu && \
    pip install --no-cache-dir fastapi==0.115.6 "uvicorn[standard]==0.34.0" pydantic==2.10.4 tokenizers==0.21.0

COPY darax_model.py darax_serve.py weights.pt tokenizer.json ./

EXPOSE 8000

CMD ["sh", "-c", "uvicorn darax_serve:app --host 0.0.0.0 --port ${PORT:-10000}"]
