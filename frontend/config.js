// InsightAI Platform Configuration
const CONFIG = {
  // API Configuration
  API: {
    BASE_URL: "http://192.28.3.186:8081/api/v1",
    
    // Document Intelligence Endpoints
    DOCUMENT_INTELLIGENCE: {
      UPLOAD_INGEST: "/documents_ingestion/upload-and-ingest",
      CHAT: "/chat/",
      STATUS: "/documents_ingestion/status"
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
  
  // AI Models Configuration
  AI_MODELS: [
    {
      id: "gpt-4-turbo",
      name: "GPT-4 Turbo",
      provider: "OpenAI",
      description: "Most capable model for complex reasoning"
    },
    {
      id: "claude-3-5-sonnet",
      name: "Claude 3.5 Sonnet",
      provider: "Anthropic",
      description: "Best for analysis and long documents"
    },
    {
      id: "gemini-pro",
      name: "Gemini Pro",
      provider: "Google",
      description: "Fast and efficient for general tasks"
    },
    {
      id: "llama-3-1-70b",
      name: "Llama 3.1 70B",
      provider: "Meta",
      description: "Open-source powerhouse"
    }
  ],
  
  // Upload Configuration
  UPLOAD: {
    // Document Intelligence
    DOCUMENT: {
      SUPPORTED_FORMATS: [".pdf", ".docx", ".txt"],
      MAX_FILE_SIZE: 10485760, // 10MB
      ACCEPTED_MIME_TYPES: [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
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
    DEFAULT_COLLECTION: "default_collection"
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