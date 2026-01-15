// InsightAI Platform Configuration
const CONFIG = {
  // API Configuration
  API: {
    BASE_URL: "http://127.0.0.1:8000/api/v1",
    
    // Document Intelligence Endpoints
    DOCUMENT_INTELLIGENCE: {
      UPLOAD_INGEST: "/ingest",           // POST - Upload & ingest PDF
      QUERY: "/query",                    // POST - Text RAG query
      QUERY_MULTIMODAL: "/query-multimodal", // POST - Multi-modal RAG query
      MODELS: "/models",                  // GET - List available LLM models
      DELETE_DOCUMENT: "/document",       // DELETE - Remove document vectors
      DOCUMENT_STATS: "/document",        // GET - /document/{id}/stats
      STATUS: "/status"                   // GET - RAG system status
    },
    
    // Code Intelligence Endpoints (Placeholder)
    CODE_INTELLIGENCE: {
      UPLOAD_REPO: "/code_ingestion/upload-repo",
      ANALYZE: "/code_intelligence/analyze",
      CHAT: "/code_intelligence/chat/"
    },
    
    // Meeting Intelligence Endpoints (Placeholder)
    MEETING_INTELLIGENCE: {
      UPLOAD_TRANSCRIPT: "/meeting_ingestion/upload-transcript",
      UPLOAD_AUDIO: "/meeting_ingestion/upload-audio",
      SUMMARIZE: "/meeting_intelligence/summarize",
      CHAT: "/meeting_intelligence/chat/"
    },
    
    // Career Intelligence Endpoints (Placeholder)
    CAREER_INTELLIGENCE: {
      UPLOAD_RESUME: "/career_ingestion/upload-resume",
      UPLOAD_JD: "/career_ingestion/upload-job-description",
      ANALYZE: "/career_intelligence/analyze",
      CHAT: "/career_intelligence/chat/"
    },
    
    TIMEOUT: 30000 // 30 seconds
  },
  
  // AI Models Configuration (from llm_factory.py)
  AI_MODELS: [
    {
      id: "groq_maverick",
      name: "Llama 4 Maverick 17B",
      provider: "Groq",
      description: "Multi-modal model supporting text and images (default)"
    },
    {
      id: "cohere_command_r_plus",
      name: "Command R+",
      provider: "Cohere",
      description: "Powerful model for complex reasoning"
    },
    {
      id: "cohere_command_a",
      name: "Command A",
      provider: "Cohere",
      description: "Latest Cohere model (2025)"
    },
    {
      id: "groq_llama3_8b",
      name: "Llama 3 8B",
      provider: "Groq",
      description: "Fast and efficient for general tasks"
    },
    {
      id: "groq_gemma_7b",
      name: "Gemma 7B",
      provider: "Groq",
      description: "Google's open model via Groq"
    },
    {
      id: "groq_mixtral_8x7b",
      name: "Mixtral 8x7B",
      provider: "Groq",
      description: "Mixture of experts model"
    }
  ],
  
  // RAG Query Defaults
  RAG: {
    DEFAULT_TOP_K: 10,           // Candidates to retrieve from vector DB
    DEFAULT_TOP_N: 3,            // Results after reranking
    DEFAULT_TEMPERATURE: 0.7,
    DEFAULT_USE_RERANKER: true,
    DEFAULT_MODEL: "groq_maverick"
  },

  // Ingestion Defaults
  INGESTION: {
    DEFAULT_CHUNK_SIZE: 312,     // Max tokens per chunk
    DEFAULT_CHUNK_OVERLAP: 50,   // Overlap tokens between chunks
    DEFAULT_EXTRACT_IMAGES: true,
    DEFAULT_EMBED_IMAGES: true,
    DEFAULT_NAMESPACE: "default"        // Pinecone namespace
  },

  // Upload Configuration
  UPLOAD: {
    // Document Intelligence (Backend only supports PDF currently)
    DOCUMENT: {
      SUPPORTED_FORMATS: [".pdf"],
      MAX_FILE_SIZE: 52428800, // 50MB for PDFs
      ACCEPTED_MIME_TYPES: [
        "application/pdf"
      ]
    },
    
    // Code Intelligence
    CODE: {
      SUPPORTED_FORMATS: [".zip", ".tar.gz"],
      MAX_FILE_SIZE: 52428800, // 50MB
      GITHUB_URL_PATTERN: /^https:\/\/github\.com\/[\w-]+\/[\w-]+/
    },
    
    // Meeting Intelligence
    MEETING: {
      TRANSCRIPT_FORMATS: [".txt", ".srt", ".vtt"],
      AUDIO_FORMATS: [".mp3", ".wav", ".m4a"],
      MAX_FILE_SIZE: 104857600 // 100MB
    },
    
    // Career Intelligence
    CAREER: {
      RESUME_FORMATS: [".pdf", ".docx"],
      JD_FORMATS: [".pdf", ".docx", ".txt"],
      MAX_FILE_SIZE: 5242880 // 5MB
    }
  },
  
  // User Configuration
  USER: {
    DEFAULT_USERNAME: "user",
    DEFAULT_NAMESPACE: "default"  // Pinecone namespace (empty = default namespace)
  },
  
  // Polling Configuration
  POLLING: {
    STATUS_CHECK_INTERVAL: 3000, // 3 seconds
    MAX_RETRIES: 100
  },
  
  // UI Configuration
  UI: {
    TOAST_DURATION: 3000,
    ERROR_TOAST_DURATION: 5000,
    ANIMATION_DURATION: 300,
    MAX_CONVERSATION_HISTORY: 50
  },
  
  // Local File Paths (for development)
  LOCAL_PATHS: {
    PDF_BASE_PATH: "/path/to/local/pdfs/",
    IMAGE_BASE_PATH: "/path/to/local/images/"
  }
};

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CONFIG;
}