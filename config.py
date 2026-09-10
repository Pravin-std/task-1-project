# Configuration file for HRBot - Open Source AI HR Assistant

import os

# Set HuggingFace and temporary cache directories to persistent location on New Volume
HF_CACHE_DIR = "/run/media/pravin/New Volume/huggingface-cache"
TMP_DIR = os.path.join(HF_CACHE_DIR, "tmp")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR
os.environ["TMPDIR"] = TMP_DIR



# Model Configuration
class ModelConfig:
    """Configuration for open-source AI models"""
    
    # Embedding Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and efficient
    EMBEDDING_DEVICE = "cpu"  # Use CPU for compatibility
    
    # LLM Model Settings - Multiple options based on system capabilities
    LLM_MODELS = {
        "lightweight": {
            "name": "google/flan-t5-small",
            "type": "text2text-generation",
            "max_length": 256,
            "description": "Very lightweight, good for low-resource systems"
        },
        "conversational": {
            "name": "microsoft/DialoGPT-medium", 
            "type": "text-generation",
            "max_length": 512,
            "description": "Better for conversational responses"
        },
        "qa_optimized": {
            "name": "distilbert-base-cased-distilled-squad",
            "type": "question-answering",
            "max_length": 384,
            "description": "Optimized for Q&A tasks"
        }
    }
    
    # Default model selection
    DEFAULT_LLM = "lightweight"  # Change to "conversational" or "qa_optimized" for better performance
    
    # Vector Database Settings
    VECTOR_DB_PATH = "vectorstore"
    COLLECTION_NAME = "hr_documents"
    SIMILARITY_SEARCH_K = 3  # Number of relevant chunks to retrieve
    
    # Text Processing Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Generation Settings
    TEMPERATURE = 0.7
    MAX_NEW_TOKENS = 256
    DO_SAMPLE = True

# Database Configuration
class DatabaseConfig:
    """Configuration for SQLite database"""
    
    DB_PATH = "data/hrbot.db"
    
    # Table schemas
    EMPLOYEE_TABLE = """
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT UNIQUE NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        joining_date DATE NOT NULL,
        salary REAL,
        manager_id TEXT,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    LEAVE_TABLE = """
    CREATE TABLE IF NOT EXISTS leave_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        leave_type TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE NOT NULL,
        days_requested INTEGER NOT NULL,
        reason TEXT,
        status TEXT DEFAULT 'pending',
        approved_by TEXT,
        approved_date TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    """
    
    PERFORMANCE_TABLE = """
    CREATE TABLE IF NOT EXISTS performance_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        reviewer_id TEXT NOT NULL,
        review_period TEXT NOT NULL,
        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
        feedback TEXT,
        goals TEXT,
        sentiment_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    """
    
    CANDIDATES_TABLE = """
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        position_applied TEXT NOT NULL,
        resume_path TEXT,
        skills TEXT,  -- JSON string of extracted skills
        experience_years INTEGER,
        match_score REAL,
        status TEXT DEFAULT 'applied',
        interview_date TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

# Application Configuration
class AppConfig:
    """General application configuration"""
    
    # Streamlit Settings
    PAGE_TITLE = "HRBot - AI HR Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    # File Upload Settings
    UPLOAD_DIR = "data"
    ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Email Settings (for notifications)
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    
    # User Roles
    USER_ROLES = ["admin", "hr", "employee"]
    
    # Leave Types
    LEAVE_TYPES = [
        "Annual Leave",
        "Sick Leave", 
        "Maternity Leave",
        "Paternity Leave",
        "Emergency Leave",
        "Unpaid Leave"
    ]
    
    # Performance Rating Scale
    PERFORMANCE_RATINGS = {
        1: "Needs Improvement",
        2: "Below Expectations", 
        3: "Meets Expectations",
        4: "Exceeds Expectations",
        5: "Outstanding"
    }

# Utility Functions
def get_model_config(model_type="default"):
    """Get model configuration based on type"""
    if model_type in ModelConfig.LLM_MODELS:
        return ModelConfig.LLM_MODELS[model_type]
    return ModelConfig.LLM_MODELS[ModelConfig.DEFAULT_LLM]

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        AppConfig.UPLOAD_DIR,
        ModelConfig.VECTOR_DB_PATH,
        "models",  # For downloaded models
        "logs"     # For application logs
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories on import
ensure_directories()
