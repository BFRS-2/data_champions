"""
Command-line interface for the Jira Tracker Bot.
"""

import typer
from rich.console import Console
import sys
import logging

from .core.bot import main as bot_main
from .config.settings import get_settings, ENV_VARS
from .config.logging_config import setup_logging
from .utils.exceptions import handle_error, ConfigurationError

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="jira",
    help="Jira Tracker Bot - Track and analyze your Jira tickets with AI",
    add_completion=False
)
console = Console()

def check_environment():
    """Check if all required environment variables are set."""
    settings = get_settings()
    missing_vars = []
    
    # Check Jira settings
    if not settings["jira"]["url"]:
        missing_vars.append(ENV_VARS["JIRA_URL"])
    if not settings["jira"]["email"]:
        missing_vars.append(ENV_VARS["JIRA_EMAIL"])
    if not settings["jira"]["api_token"]:
        missing_vars.append(ENV_VARS["JIRA_API_TOKEN"])
    
    # Check OpenAI settings
    if not settings["openai"]["api_key"]:
        missing_vars.append(ENV_VARS["OPENAI_API_KEY"])
    
    # Check Telegram settings
    if not settings["telegram"]["bot_token"]:
        missing_vars.append(ENV_VARS["TELEGRAM_BOT_TOKEN"])
    
    if missing_vars:
        console.print("[red]Missing required environment variables:[/red]")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\n[yellow]Please create a .env file with these variables:[/yellow]")
        console.print("""
# Jira Configuration
JIRA_URL=your-jira-instance.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-jira-api-token

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
        """)
        raise ConfigurationError("Missing required environment variables")
    return True

@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    config: str = typer.Option(None, "--config", "-c", help="Path to config file")
):
    """Start the Jira Tracker Bot."""
    try:
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        check_environment()
        logger.info("Starting Jira Tracker Bot...")
        console.print("[green]🚀 Starting Jira Tracker Bot...[/green]")
        bot_main()
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        console.print(f"[red]Error: Missing required package. Please run:[/red]")
        console.print("pip install -e .")
        raise typer.Exit(1)
    except Exception as e:
        error_message = handle_error(e)
        logger.error(error_message, exc_info=True)
        console.print(f"[red]{error_message}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 