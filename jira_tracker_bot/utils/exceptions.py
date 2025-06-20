class JiraTrackerError(Exception):
    """Base exception for Jira Tracker Bot."""
    pass

class ConfigurationError(JiraTrackerError):
    """Raised when there's an error in configuration."""
    pass

class JiraAPIError(JiraTrackerError):
    """Raised when there's an error with Jira API."""
    pass

class OpenAIError(JiraTrackerError):
    """Raised when there's an error with OpenAI API."""
    pass

class TelegramError(JiraTrackerError):
    """Raised when there's an error with Telegram API."""
    pass

class ValidationError(JiraTrackerError):
    """Raised when input validation fails."""
    pass

class CacheError(JiraTrackerError):
    """Raised when there's an error with caching."""
    pass

def handle_error(error: Exception) -> str:
    """Handle different types of errors and return appropriate messages."""
    if isinstance(error, ConfigurationError):
        return f"Configuration error: {str(error)}"
    elif isinstance(error, JiraAPIError):
        return f"Jira API error: {str(error)}"
    elif isinstance(error, OpenAIError):
        return f"OpenAI API error: {str(error)}"
    elif isinstance(error, TelegramError):
        return f"Telegram API error: {str(error)}"
    elif isinstance(error, ValidationError):
        return f"Validation error: {str(error)}"
    elif isinstance(error, CacheError):
        return f"Cache error: {str(error)}"
    else:
        return f"Unexpected error: {str(error)}" 