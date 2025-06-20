import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Environment variables
ENV_VARS = {
    "JIRA_URL": "JIRA_URL",
    "JIRA_EMAIL": "JIRA_EMAIL",
    "JIRA_API_TOKEN": "JIRA_API_TOKEN",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "TELEGRAM_BOT_TOKEN": "TELEGRAM_BOT_TOKEN",
}

# Jira settings
JIRA_SETTINGS = {
    "url": os.getenv(ENV_VARS["JIRA_URL"]),
    "email": os.getenv(ENV_VARS["JIRA_EMAIL"]),
    "api_token": os.getenv(ENV_VARS["JIRA_API_TOKEN"]),
    "max_results": 50,
    "timeout": 30,
}

# OpenAI settings
OPENAI_SETTINGS = {
    "api_key": os.getenv(ENV_VARS["OPENAI_API_KEY"]),
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000,
}

# Telegram settings
TELEGRAM_SETTINGS = {
    "bot_token": os.getenv(ENV_VARS["TELEGRAM_BOT_TOKEN"]),
    "poll_interval": 1.0,
    "timeout": 30,
}

# Data storage settings
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Cache settings
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        "jira": JIRA_SETTINGS,
        "openai": OPENAI_SETTINGS,
        "telegram": TELEGRAM_SETTINGS,
        "data_dir": str(DATA_DIR),
        "cache_dir": str(CACHE_DIR),
        "cache_expiry": CACHE_EXPIRY,
    } 