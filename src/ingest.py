import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_vector_store

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, os.getenv("PDF_PATH", "document.pdf"))  # relativo à raiz do projeto
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


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
    store.add_documents(documents=chunks, ids=ids)
    print(f"Ingestão concluída: {len(documents)} páginas, {len(chunks)} chunks salvos.")


if __name__ == "__main__":
    ingest_pdf()
