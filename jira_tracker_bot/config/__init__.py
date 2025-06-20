"""
Configuration settings for the Jira Tracker Bot.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Default settings
DEFAULT_SETTINGS = {
    "notification_preferences": {
        "mentions": True,
        "stale_tickets": True,
        "priority_changes": True,
        "status_changes": True
    },
    "stale_threshold_days": 7,
    "update_interval_minutes": 30
}

# Environment variable names
ENV_VARS = {
    "JIRA_URL": "JIRA_URL",
    "JIRA_EMAIL": "JIRA_EMAIL",
    "JIRA_API_TOKEN": "JIRA_API_TOKEN",
    "OPENAI_API_KEY": "OPENAI_API_KEY"
} 