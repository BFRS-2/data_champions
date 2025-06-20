"""
Validation utilities for the Jira Tracker Bot.
"""

import re
from urllib.parse import urlparse

def validate_jira_url(url: str) -> bool:
    """Validate Jira URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and (
            result.scheme in ['http', 'https'] and
            any(domain in result.netloc.lower() for domain in ['atlassian.net', 'jira'])
        )
    except:
        return False

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) 