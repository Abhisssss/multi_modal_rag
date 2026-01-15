# Multi Modal RAG System

A production-ready Retrieval Augmented Generation (RAG) system built with FastAPI, Pinecone, and multi-modal embedding capabilities.

## Quick Setup

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multi-modal-rag
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   GROQ_API_KEY = "your-groq-api-key"
   COHERE_API_KEY = "your-cohere-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_ENV = "your-pinecone-environment"
   PINECONE_INDEX = "your-pinecone-index-name"
   PINECONE_REGION = "your-pinecone-region"
   PINECONE_NAMESPACE = "your-namespace"
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the API**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Frontend

Open `frontend/insightai-main.html` directly in your browser to access the web interface.

---

## Architecture Overview

![RAG System Architecture](demo/RAG%20system%20architecture.png)

### Demo

https://github.com/user-attachments/assets/demo/RAG_VideoDemo.mp4

[View Demo Video](demo/RAG_VideoDemo.mp4)

### Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: Pinecone
- **LLM Providers**: Cohere (Command R+, Command A), Groq (Llama), OpenAI (OSS 120b)
- **Embedding & Reranking**: Cohere Embed v4 & Cohere Rerank
- **Orchestration**: Custom Python Framework (No LangChain)

---

## Productionization & Scalability: My Thoughts

To actually productionize this or deploy it on a hyper-scaler like AWS or GCP, I need to understand the exact requirements first. What types of documents are we processing? Is it just PDFs, or are we looking at Docx and PPTs? What's the average page count?

Here is how I would approach it:

- **Database Strategy**: Let's face it, it's not just about the Vector DB. We need a relational database to maintain user collections, file metadata, ingestion logs, and user info. Unless we use something like Oracle DB 23ai (which has inbuilt capabilities for both relational and vector data), we need to maintain a connection to both.

- **Monolith vs. Microservices**: My current solution uses FastAPI. If we go the microservices route, we waste time connecting and disconnecting from the DB. In a monolithic structure, we can use DB pools to maintain hundreds of parallel connections open all the time.

- **Ingestion Costs**: If we use services like AWS Textract, we will burn through money. I prefer experimenting with open-source models like LLMSherpa, PyMuPDF4LLM, or Dolphin. They are amazing for extracting layout structure and headings without the high cloud cost.

- **Deployment**: If we have high parallel users, I'd dockerize this and put it on a Kubernetes cluster. For occasional high traffic, I'd stick to microservices with a queue system.

---

## RAG/LLM Approach & Decisions

### The Tech Stack

I didn't just pick random models; every model here was carefully selected because they are best-in-class for enterprise RAG:

- **Embeddings**: I used Cohere Embed v4. It handles both images and text in the same dimensions, which sets us up for multi-modal capabilities later.

- **Reranking**: I'm using Cohere Rerank. Fetching top chunks via cosine similarity isn't enough; the reranker optimizes the results and drastically improves accuracy.

- **Vector DB**: I chose Pinecone. They are using gRPC protobuf from Google to give that ultra-millisecond capability for upserting and querying. I also implemented batch embedding and storing, handling 50 chunks at a time to optimize throughput.

### Why No LangChain?

For orchestration, I didn't use LangChain. It's too heavy. It might be good for building a prototype in a few hours, but it's not production-grade.

- **The Issue**: We faced a massive issue in the past where LangChain updates would break built-in function names and conventions. Maintaining it became a nightmare.

- **My Solution**: I built a custom framework in pure Python. It allows me to optimize the context window and prompt management exactly how I need it without the abstraction overhead.

### Context & Guardrails

- **Retrieval**: I fetch the top relevant chunks using dense vector search, then filter the top 3 using the Reranker.

- **Guardrails**: I use Pydantic schemas everywhere to ensure perfect structure. I also have prompt-level guardrails to handle hallucinations—basically making sure the LLM says "I don't know" if the context isn't there.

---

## Engineering Standards

- **Project Structure**: The whole project is carefully crafted into modules. `core_services` contains all the boilerplate and client code (LLM, Pinecone, Parsers). The data ingestion module is built so it can be enhanced without breaking the retrieval logic.

- **FastAPI**: I chose this because it handles requests asynchronously. It's fast compared to other Python frameworks.

- **Testing**: I didn't get a chance to write full unit tests, but I included a specific endpoint in Swagger to check the "Health" of individual components (is the LLM working? is Pinecone reachable?). This helps validate features individually.

---

## How I Used AI Tools

I absolutely used AI tools (Gemini CLI and Claude Code), but I used them carefully.

- **The "Junior Developer" Approach**: I treat the AI like a junior developer. If I just say "add a caching feature," it won't understand the nuance. I have to say: "Utilize Redis, here is the location, add a Time-To-Live, and check this condition before retrieving."

- **Planning**: I used Gemini CLI to brainstorm the initial architecture and project structure.

- **Verification**: There is a 40% chance the AI doesn't understand the requirement or uses outdated libraries (since training data cuts off). I always ask for a "plan" first, review it, and then manually verify the code—especially for things like Pinecone or Embedding clients where versions change often.

---

## What I'd Do With More Time

- **Unified System**: I noticed that all 4 assignment options (Docs, Code, Meeting, Resume) are essentially RAG problems. With more time, I would integrate all 4 into a single web interface.

- **Agentic RAG**: I would implement an Agentic feature where the system doesn't just retrieve data but can "reason." It could fetch extra info via Google Search and iterate multiple times to answer complex queries.

- **Observability**: I would build a proper observability layer using Phoenix Arize to calculate answer quality and retrieval relevance.

- **Full Async**: I would ensure every single function in the pipeline is asynchronous to handle massive concurrency.
