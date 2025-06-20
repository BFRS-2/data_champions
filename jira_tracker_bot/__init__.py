"""
Jira Tracker Bot - An AI-powered Jira ticket tracking and analysis tool.
"""

__version__ = "1.0.0"

from .core.bot import JiraTrackerBot
from .core.jira_client import JiraClient
from .core.openai_client import OpenAIClient

__all__ = ['JiraTrackerBot', 'JiraClient', 'OpenAIClient'] 