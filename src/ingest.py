import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_vector_store

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, os.getenv("PDF_PATH", "document.pdf"))  # relativo à raiz do projeto
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 20  # lotes pequenos para caber no limite de tokens/minuto do nível gratuito
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 60


def is_rate_limit(error):
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def add_in_batches(store, chunks, ids):
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(documents=chunks[start:end], ids=ids[start:end])
                break
            except Exception as error:
                if not is_rate_limit(error) or attempt == MAX_RETRIES:
                    raise
                print(f"Limite de requisições atingido; aguardando {RETRY_WAIT_SECONDS} s (tentativa {attempt}).")
                time.sleep(RETRY_WAIT_SECONDS)
        print(f"{min(end, len(chunks))}/{len(chunks)} chunks gravados.")


def ingest_pdf():
    if not PDF_PATH or not os.path.isfile(PDF_PATH):
        raise SystemExit(f"PDF_PATH={PDF_PATH!r} inválido: esperado caminho de um arquivo PDF existente")

    documents = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise SystemExit("Nenhum texto extraído do PDF.")

    # Mantém só metadados simples (evita valores vazios/complexos no JSONB)
    chunks = [
        chunk.model_copy(update={"metadata": {k: v for k, v in chunk.metadata.items() if v not in ("", None)}})
        for chunk in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(chunks))]  # ids fixos: reexecutar sobrescreve em vez de duplicar

    try:
        store = get_vector_store()
    except ValueError as error:
        raise SystemExit(f"Erro de configuração: {error}") from None
    add_in_batches(store, chunks, ids)
    print(f"Ingestão concluída: {len(documents)} páginas, {len(chunks)} chunks salvos.")


if __name__ == "__main__":
    ingest_pdf()
