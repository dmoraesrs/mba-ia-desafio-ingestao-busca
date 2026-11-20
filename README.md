# Ingestão e Busca Semântica com LangChain e Postgres

Solução do desafio do módulo **Introdução ao LangChain** (MBA em Engenharia de Software com IA, Full Cycle).

O software lê um PDF, divide o texto em chunks, gera os embeddings e grava os vetores no PostgreSQL com pgVector. Depois, um chat no terminal responde perguntas **usando apenas o conteúdo do PDF**: o que não está no documento recebe a resposta "Não tenho informações necessárias para responder sua pergunta."

## Como funciona

**Ingestão (`src/ingest.py`)**

1. Carrega o PDF com `PyPDFLoader`.
2. Divide em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`).
3. Gera o embedding de cada chunk e grava no pgVector (`PGVector`). Os chunks recebem ids fixos (`doc-0`, `doc-1`, ...), então rodar a ingestão de novo sobrescreve em vez de duplicar.

**Busca (`src/search.py`) e chat (`src/chat.py`)**

1. Vetoriza a pergunta.
2. Busca os **10 trechos mais relevantes** com `similarity_search_with_score(query, k=10)`.
3. Monta o prompt do enunciado com os trechos concatenados como CONTEXTO e chama a LLM.
4. Mostra a resposta no terminal.

## Provedores e modelos

Funciona com **OpenAI** ou **Google Gemini**. O provedor sai de `LLM_PROVIDER` no `.env`; se ficar vazio, usa OpenAI quando `OPENAI_API_KEY` estiver preenchida e Gemini caso contrário.

| Provedor | Embeddings (padrão) | LLM (padrão) |
|----------|---------------------|--------------|
| OpenAI | `text-embedding-3-small` | `gpt-5-nano` |
| Gemini | `models/gemini-embedding-001` | `gemini-2.5-flash-lite` |

Os modelos são configuráveis pelas variáveis `*_EMBEDDING_MODEL` e `*_LLM_MODEL`. Nomes de modelo mudam com frequência: confira a documentação do provedor se algum for descontinuado.

> **Trocou o modelo de embeddings?** Cada modelo gera vetores com uma dimensão diferente e a tabela é criada na primeira ingestão. Antes de ingerir com outro modelo, apague o volume do banco (`docker compose down -v`) ou use outro `PG_VECTOR_COLLECTION_NAME`.

## Pré-requisitos

- Python 3.10 ou superior
- Docker e Docker Compose
- Uma API key da OpenAI ou do Google (Gemini)

## Como executar

**1. Criar e ativar o ambiente virtual e instalar as dependências**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Configurar as variáveis de ambiente**

```bash
cp .env.example .env
```

Edite o `.env` e preencha a chave do provedor escolhido (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`). Os demais valores já vêm prontos para o banco do `docker-compose.yml`:

| Variável | Valor padrão | Descrição |
|----------|--------------|-----------|
| `LLM_PROVIDER` | vazio | `openai` ou `google` (vazio = detecta pela chave) |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Conexão com o Postgres |
| `PG_VECTOR_COLLECTION_NAME` | `documentos` | Nome da collection no pgVector |
| `PDF_PATH` | `document.pdf` | PDF a ingerir, relativo à raiz do projeto |

**3. Subir o banco de dados**

```bash
docker compose up -d
```

**4. Executar a ingestão do PDF**

```bash
python src/ingest.py
```

Saída esperada: `Ingestão concluída: 34 páginas, 67 chunks salvos.`

**5. Rodar o chat**

```bash
python src/chat.py
```

Digite `sair` (ou Ctrl+C) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: O faturamento da Alfa Energia S.A. é de R$ 722.875.391,46.

PERGUNTA: Em que ano foi fundada a Aliança Esportes ME?
RESPOSTA: A Aliança Esportes ME foi fundada em 2002.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Estrutura do projeto

```
├── docker-compose.yml    # Postgres 17 + pgVector
├── requirements.txt      # Dependências (versões fixadas, sem vulnerabilidades conhecidas)
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Provedores, vector store, prompt e busca
│   └── chat.py           # CLI de perguntas e respostas
├── document.pdf          # PDF para ingestão
└── README.md
```

## Problemas comuns

| Sintoma | Causa e solução |
|---------|-----------------|
| `OPENAI_API_KEY não definida no .env` | Faltou copiar o `.env.example` para `.env` ou preencher a chave |
| `different vector dimensions` na ingestão | O modelo de embeddings mudou: `docker compose down -v`, `docker compose up -d` e ingira de novo |
| `connection refused` na porta 5432 | O banco não está de pé (`docker compose ps`) ou outra aplicação ocupa a porta 5432 |
