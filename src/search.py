import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector

load_dotenv()

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_provider():
    """Usa LLM_PROVIDER se definido; senão escolhe pela chave presente no .env."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "google"
    if provider not in ("openai", "google"):
        raise ValueError(f"LLM_PROVIDER={provider!r} inválido: esperado 'openai' ou 'google'")
    key_var = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"
    if not os.getenv(key_var):
        raise ValueError(f"{key_var} não definida no .env")
    return provider


def get_embeddings():
    if get_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"))


def get_llm():
    if get_provider() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"))

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"), temperature=0)


def get_vector_store():
    for var in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
        if not os.getenv(var):
            raise ValueError(f"{var} não definida no .env")
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def search_prompt(question=None):
    """Monta a cadeia de busca. Retorna uma função pergunta -> resposta, ou None se falhar."""
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as error:
        print(f"Erro ao inicializar a busca: {error}")
        return None

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    def ask(pergunta):
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        return chain.invoke({"contexto": contexto, "pergunta": pergunta}).content.strip()

    if question:
        return ask(question)
    return ask
