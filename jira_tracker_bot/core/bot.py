import os
import logging
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import requests
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import signal # Added for graceful shutdown
import asyncio # Added for graceful shutdown
import re
from typing import Optional, Dict
import tempfile
import telegram

# Import the new clients
from ..core.openai_client import OpenAIClient
from ..core.jira_client import JiraClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Jira configuration (kept here as env vars are loaded here)
JIRA_DOMAIN = os.getenv('JIRA_URL')
JIRA_EMAIL = os.getenv('JIRA_EMAIL')
JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

PROACTIVE_CHECK_INTERVAL_SECONDS = 3600 # 1 hour for production, can be lower for testing

class JiraTrackerBot:
    def __init__(self):
        # Ensure environment variables are loaded before initializing clients
        if not all([TELEGRAM_BOT_TOKEN, JIRA_DOMAIN, JIRA_EMAIL, JIRA_API_TOKEN, OPENAI_API_KEY]):
            missing_vars = [var for var in ['TELEGRAM_BOT_TOKEN', 'JIRA_URL', 'JIRA_EMAIL', 'JIRA_API_TOKEN', 'OPENAI_API_KEY'] if not os.getenv(var)]
            raise ValueError(f"Missing required environment variables: {missing_vars}. Please check your .env file.")

        self.telegram_token = TELEGRAM_BOT_TOKEN
        self.jira_client = JiraClient(jira_domain=JIRA_DOMAIN, jira_email=JIRA_EMAIL, jira_api_token=JIRA_API_TOKEN)
        self.openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
        
        # Initialize the application
        self.application = Application.builder().token(self.telegram_token).build()

        # Store user data for proactive notifications
        self.conversation_history = {}
        self._user_data = {}
        self._user_last_checked_timestamp = {}
        self.ticket_cache = {}
        self._notification_preferences = {}
        self._pending_comment = {}  # Store pending comment issue_key for reply-to-add-comment flow
        self._last_ticket_context = {}  # Store last ticket key for intent chaining
        self._status_mapping = {
            'qa': 'QA',
            'done': 'Done',
            'staging': 'Staging',
            'in progress': 'In Progress',
            'to do': 'To Do',
            'blocked': 'Blocked'
        }
        self._thread_context = {}  # Store thread-aware context for each user
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(CommandHandler("my_open_tickets", self.my_open_tickets))
        self.application.add_handler(CommandHandler("my_reported_tickets", self.my_reported_tickets))
        # Unified callback handler
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_queries))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message_wrapper))

        # Add new handlers for AI features
        self.application.add_handler(CommandHandler("analyze_ticket", self.analyze_ticket))
        self.application.add_handler(CommandHandler("workload", self.analyze_workload))
        self.application.add_handler(CommandHandler("auto_assign", self.auto_assign_ticket))
        self.application.add_handler(CommandHandler("predict", self.predict_ticket_metrics))
        self.application.add_handler(CommandHandler("auto_create_issues", self.auto_create_issues))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_prd_file))

        # Initialize job queue for scheduled tasks
        self.job_queue = self.application.job_queue
        self.job_queue.run_repeating(self.check_for_mentions_and_notify, interval=PROACTIVE_CHECK_INTERVAL_SECONDS, first=0)
        self.job_queue.run_repeating(self.check_for_stale_tickets_and_notify, interval=PROACTIVE_CHECK_INTERVAL_SECONDS, first=0)

    def extract_issue_key(self, text):
        """Extracts a Jira issue key from the given text, handling both 'ABC-123' and 'ABC 123' formats."""
        # Try to find ABC-123 pattern
        match = re.search(r'([A-Z][A-Z0-9]+)-([0-9]+)', text, re.IGNORECASE)
        if match:
            return match.group(0).upper().rstrip(':.-_ ')
        # Try to find ABC 123 pattern and convert to ABC-123
        match = re.search(r'([A-Z][A-Z0-9]+)[\s_]+([0-9]+)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()}-{match.group(2)}".rstrip(':.-_ ')
        return None

    def get_jira_issue(self, issue_key: str) -> Optional[Dict]:
        """Get Jira issue details."""
        try:
            logger.debug(f"Fetching Jira issue: {issue_key}")
            issue, error = self.jira_client.get_jira_issue(issue_key)
            logger.debug(f"Jira API response for {issue_key}: {issue}")
            if error:
                logger.error(f"Error fetching Jira issue {issue_key}: {error}")
                return None
            return issue
        except Exception as e:
            logger.error(f"Error fetching Jira issue {issue_key}: {e}")
            return None

    def get_my_jira_profile(self):
        """Fetches the bot's Jira profile to get its email and display name."""
        return self.jira_client.get_my_jira_profile()

    def generate_ai_error_message(self, error):
        """Generate user-friendly error messages using AI (via OpenAIClient)."""
        return self.openai_client.generate_user_friendly_error(error)

    def get_smart_summary(self, issue_data):
        """Generate AI-powered summary of the issue (via OpenAIClient)."""
        description = issue_data['fields'].get('description', '')
        summary = issue_data['fields'].get('summary', '')
        return self.openai_client.generate_smart_summary(summary, description)

    def analyze_comments(self, comments):
        """Analyze comments using AI to extract key points (via OpenAIClient)."""
        if not comments or not comments.get('comments'):
            return "No comments to analyze."
        comment_texts = [comment['body']['content'][0]['content'][0]['text'] 
                           for comment in comments['comments']]
        return self.openai_client.analyze_comments_ai(comment_texts)

    def find_similar_tickets(self, issue_data):
        """Find similar tickets using Jira API and text similarity."""
        # First, get relevant tickets from Jira via JiraClient
        all_project_tickets = self.jira_client.find_similar_tickets(issue_data) # This method already filters the current issue

        # Then, apply TF-IDF similarity if there are enough tickets
        current_ticket_text = f"{issue_data['fields']['summary']} {issue_data['fields'].get('description', '')}"
        ticket_texts = [f"{ticket['fields']['summary']} {ticket['fields'].get('description', '')}" 
                        for ticket in all_project_tickets]

        if not ticket_texts:
            return [] # No other tickets to compare with

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([current_ticket_text] + ticket_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

            # Get top 3 similar tickets from the *filtered_tickets*
            similar_indices = similarities.argsort()[-3:][::-1]
            
            # Filter out indices that are out of bounds or negative
            valid_similar_indices = [i for i in similar_indices if 0 <= i < len(all_project_tickets)]
            similar_tickets = [all_project_tickets[i] for i in valid_similar_indices]

            return similar_tickets
        except Exception as e:
            logger.error(f"Unexpected error during TF-IDF similarity calculation: {e}")
            return []

    async def check_for_mentions_and_notify(self, context: ContextTypes.DEFAULT_TYPE):
        """Scheduled task to check for new mentions in Jira comments and notify users."""
        logger.info("Running scheduled mention check...")
        for user_id, user_data in self._user_data.items():
            chat_id = user_data['chat_id']
            jira_email = user_data['jira_email']
            jira_display_name = user_data['jira_display_name']

            # Check if user has enabled mention notifications
            if not self._notification_preferences.get(user_id, {}).get('mentions', True): # Default to True
                logger.info(f"Mention notifications disabled for user {user_id}. Skipping.")
                continue

            try:
                # Get all issues assigned to or reported by the user
                # This JQL aims to cover tickets the user is likely to be involved in
                jql_query = f"assignee = \"{jira_email}\" OR reporter = \"{jira_email}\" ORDER BY updated DESC"
                issues, error = self.jira_client.search_issues_by_jql(jql_query)
                if error:
                    logger.error(f"Error fetching issues for mention check for user {user_id}: {error}")
                    continue

                for issue in issues:
                    comments, error = self.jira_client.get_comments(issue['key'])
                    if error:
                        logger.error(f"Error fetching comments for issue {issue['key']} during mention check: {error}")
                        continue

                    if comments:
                        # Check for mentions and new comments
                        for comment in comments:
                            if self._user_last_checked_timestamp.get(user_id) and comment['created'] > self._user_last_checked_timestamp[user_id]:
                                if jira_display_name and f"@{jira_display_name}" in comment['body'] or (jira_email and f"@{jira_email.split('@')[0]}" in comment['body']):
                                    # New mention! Get sentiment
                                    sentiment = self.openai_client.analyze_sentiment(comment['body'])
                                    notification_message = (
                                        f"🔔 *New Mention in Jira!*\n\n"
                                        f"*Issue:* [{issue['key']}] - {issue['fields']['summary']}\n"
                                        f"*Comment by:* {comment['author']['displayName']}\n"
                                        f"*Sentiment:* {sentiment}\n\n"
                                        f"*{comment['body']}*"
                                    )
                                    await self.send_message(chat_id, notification_message)
                                    # Update last checked timestamp for this user after sending notification
                                    self._user_last_checked_timestamp[user_id] = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                logger.error(f"Error in scheduled mention check for user {user_id}: {e}")

    async def check_for_stale_tickets_and_notify(self, context: ContextTypes.DEFAULT_TYPE):
        """Scheduled task to check for stale open tickets assigned to the user."""
        logger.info("Running scheduled check for stale tickets...")
        for user_id, user_data in self._user_data.items():
            chat_id = user_data['chat_id']
            jira_email = user_data['jira_email']

            # Check if user has enabled stale ticket notifications
            if not self._notification_preferences.get(user_id, {}).get('stale_tickets', True): # Default to True
                logger.info(f"Stale ticket notifications disabled for user {user_id}. Skipping.")
                continue

            try:
                # JQL to find open tickets assigned to the user, not updated in the last 24 hours
                # This means 'updated <= -1d'
                jql_query = f"assignee = \"{jira_email}\" AND statusCategory != \"Done\" AND updated <= \"-1d\" ORDER BY updated ASC"
                stale_issues, error = self.jira_client.search_issues_by_jql(jql_query)
                if error:
                    logger.error(f"Error fetching stale issues for user {user_id}: {error}")
                    continue

                if stale_issues:
                    notification_message = "🐢 *Stale Jira Tickets Assigned to You!*\n\n"
                    notification_message += "The following tickets assigned to you haven't been updated in over 24 hours:\n\n"
                    for issue in stale_issues[:5]: # Limit to top 5 stale tickets
                        response_text = (
                            f"*[{issue['key']}]* - {issue['fields']['summary']} "
                            f"(Status: {issue['fields']['status']['name']})\n"
                        )
                        notification_message += response_text
                    notification_message += "\nConsider checking these to ensure progress!"
                    await self.send_message(chat_id, notification_message)

            except Exception as e:
                logger.error(f"Error in scheduled stale ticket check for user {user_id}: {e}")

    async def send_message(self, chat_id: int, text: str, reply_markup: InlineKeyboardMarkup = None) -> None:
        """Send a message to the specified chat."""
        try:
            # Check if message is too long (Telegram's limit is 4096 characters)
            if len(text) > 4000:  # Using 4000 to be safe
                # Create a temporary file with the message content
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"jira_response_{timestamp}.txt"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Send the file
                with open(filename, 'rb') as f:
                    await self.application.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        caption="📄 Response is too long for a single message. Here's the complete response in a file."
                    )
                
                # Clean up the temporary file
                os.remove(filename)
            else:
                # Send as normal message if within limits
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode='Markdown',
                    reply_markup=reply_markup
                )
        except Exception as e:
            logger.error(f"Error sending message to chat_id {chat_id}: {e}")
            # Try sending a simpler message if the original fails
            try:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text="❌ An error occurred while sending the message. Please try again."
                )
            except Exception as e2:
                logger.error(f"Error sending error message to chat_id {chat_id}: {e2}")

    async def _handle_message_wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message_text = update.message.text
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        # Check if awaiting project key for issue creation
        if self._thread_context.get(user_id, {}).get('awaiting_project_key'):
            self._thread_context[user_id]['pending_project_key'] = message_text.strip().upper()
            self._thread_context[user_id]['awaiting_project_key'] = False
            # Now proceed to create issues if plan is present
            plan = self._thread_context[user_id].get('pending_plan')
            if plan:
                results = self.create_jira_issues_from_plan(plan, self._thread_context[user_id]['pending_project_key'])
                await self.send_message(chat_id, f"✅ Created issues:\n{results}")
                self._thread_context[user_id]['pending_plan'] = None
                self._thread_context[user_id]['pending_project_key'] = None
            else:
                await self.send_message(chat_id, "❌ No plan found to create issues.")
            return

        # Store user data for proactive notifications
        if user_id not in self._user_data:
            self._user_data[user_id] = {'chat_id': chat_id}
            # Initialize notification preferences for new user
            self._notification_preferences[user_id] = {
                'mentions': True,
                'stale_tickets': True
            }
            # Get Jira profile for the user (only for new users)
            jira_profile_data, jira_profile_error = self.jira_client.get_my_jira_profile()
            if jira_profile_data:
                self._user_data[user_id]['jira_email'] = jira_profile_data.get('emailAddress', JIRA_EMAIL)
                self._user_data[user_id]['jira_display_name'] = jira_profile_data.get('displayName', '')
                logger.info(f"User {user_id} registered with Jira email: {self._user_data[user_id]['jira_email']}")
            else:
                logger.warning(f"Could not fetch Jira profile for user {user_id}. Mentions might not work correctly. Error: {jira_profile_error}. Ensure JIRA_EMAIL in .env is correct.")
                self._user_data[user_id]['jira_email'] = JIRA_EMAIL # Fallback
                self._user_data[user_id]['jira_display_name'] = "" # Fallback

        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        self.conversation_history[user_id].append({"role": "user", "content": message_text})

        await self.handle_message(message_text, chat_id, user_id)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        welcome_message = """👋 *Hello! I'm your Jira Tracker Bot.*\n\n
I can help you get information about Jira tickets, add comments, and change statuses.\n\n
*Here's what I can do:*
- Ask me about a ticket: `What is the status of CBT-123?`
- Get comments: `Show comments on CBT-123`
- Add comments: `Add comment to CBT-123: This is my comment.`
- Change status: `Change status of CBT-123 to In Progress`
- Search tickets: `Show me all open bugs in Project X`
- See your open tickets: `/my_open_tickets`
- See tickets you reported: `/my_reported_tickets`
- Manage notifications: `/settings`\n\n
Type /help to see all commands."""
        await self.send_message(update.effective_chat.id, welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /help is issued."""
        help_message = (
            "*Jira Tracker Bot Help:*\n\n"
            "*Basic Commands:*\n"
            "- `/start`: Start interaction and see a welcome message\n"
            "- `/help`: Show this help message\n"
            "- `/my_open_tickets`: List all Jira tickets currently assigned to you\n"
            "- `/my_reported_tickets`: List all Jira tickets you have reported\n"
            "- `/settings`: Manage your notification preferences\n\n"
            "*AI-Powered Features:*\n"
            "- `/analyze_ticket <ticket_key>`: Get comprehensive AI analysis of a ticket\n"
            "- `/workload`: Analyze team workload distribution and get optimization suggestions\n"
            "- `/auto_assign <ticket_key>`: Get AI-suggested assignee for a ticket\n"
            "- `/predict <ticket_key>`: Get AI predictions for ticket metrics\n\n"
            "*Natural Language Queries:*\n"
            "- `What is the status of CBT-123?`\n"
            "- `Show comments on PROJ-456`\n"
            "- `Add comment to TASK-789: Please review this.`\n"
            "- `Change status of BUG-101 to Done`\n"
            "- `Search for all high priority tasks in Project Alpha`\n"
            "- `Summarize JIRA-200`\n\n"
            "*Proactive Notifications:*\n"
            "- Mentions in Jira comments\n"
            "- Stale ticket alerts\n"
            "- Workload imbalance warnings\n"
            "- Priority change notifications\n\n"
            "*Got questions or feedback? Feel free to ask!*"
        )
        await self.send_message(update.effective_chat.id, help_message)

    async def my_open_tickets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows all open tickets assigned to the user."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        jira_email = self._user_data.get(user_id, {}).get('jira_email')

        if not jira_email:
            await self.send_message(chat_id, "❌ Your Jira email is not set up. Please ensure JIRA_EMAIL is configured in .env.")
            return

        processing_message = await self.send_message(chat_id, "🔍 Fetching your open tickets...")
        
        jql_query = f"assignee = \"{jira_email}\" AND statusCategory != \"Done\" ORDER BY updated DESC"
        issues, error = self.jira_client.search_issues_by_jql(jql_query)

        if error:
            await self.send_message(chat_id, f"❌ Error fetching your open tickets: {error}")
            return

        if issues:
            response_text = "✅ *Your Open Jira Tickets:*\n\n"
            for issue in issues[:10]: # Limit to top 10 for brevity
                response_text += (
                    f"*[ {issue['key']}]* - {issue['fields']['summary']} "
                    f"(Status: {issue['fields']['status']['name']})\n"
                )
            await self.send_message(chat_id, response_text)
        else:
            await self.send_message(chat_id, "🎉 You have no open tickets! Keep up the great work.")

    async def my_reported_tickets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows all tickets reported by the user."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        jira_email = self._user_data.get(user_id, {}).get('jira_email')

        if not jira_email:
            await self.send_message(chat_id, "❌ Your Jira email is not set up. Please ensure JIRA_EMAIL is configured in .env.")
            return

        processing_message = await self.send_message(chat_id, "🔍 Fetching tickets you reported...")

        jql_query = f"reporter = \"{jira_email}\" ORDER BY created DESC"
        issues, error = self.jira_client.search_issues_by_jql(jql_query)

        if error:
            await self.send_message(chat_id, f"❌ Error fetching tickets you reported: {error}")
            return

        if issues:
            response_text = "✅ *Jira Tickets You Reported:*\n\n"
            for issue in issues[:10]: # Limit to top 10 for brevity
                response_text += (
                    f"*[ {issue['key']}]* - {issue['fields']['summary']} "
                    f"(Status: {issue['fields']['status']['name']})\n"
                )
            await self.send_message(chat_id, response_text)
        else:
            await self.send_message(chat_id, "ℹ️ You haven't reported any tickets yet.")

    # New command handler for settings
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sends a message with inline keyboard to manage notification settings."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        mentions_enabled = self._notification_preferences.get(user_id, {}).get('mentions', True)
        stale_tickets_enabled = self._notification_preferences.get(user_id, {}).get('stale_tickets', True)

        keyboard = [
            [InlineKeyboardButton(
                f"Mentions: {'✅ Enabled' if mentions_enabled else '❌ Disabled'}",
                callback_data=f"toggle_setting:mentions"
            )],
            [InlineKeyboardButton(
                f"Stale Tickets: {'✅ Enabled' if stale_tickets_enabled else '❌ Disabled'}",
                callback_data=f"toggle_setting:stale_tickets"
            )],
            [InlineKeyboardButton("Back to Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await self.send_message(
            chat_id,
            "⚙️ *Notification Settings:*\n\nChoose which notifications you'd like to receive.",
            reply_markup=reply_markup
        )

    async def handle_action_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle confirmation for actions like adding comments or changing status."""
        query = update.callback_query
        await query.answer()
        chat_id = query.message.chat_id

        data = query.data
        if data.startswith("confirm_action:"):
            try:
                action_data_str = data.replace("confirm_action:", "")
                action_data = json.loads(action_data_str)

                action_type = action_data.get('action')
                issue_key = action_data.get('issue_key')

                if action_type == 'add_comment':
                    comment = action_data.get('comment')
                    success, error = self.jira_client.add_comment_to_issue(issue_key, comment)
                    if success:
                        await self.send_message(chat_id, f"✅ Comment added to {issue_key}.")
                    else:
                        await self.send_message(chat_id, f"❌ Failed to add comment: {error}")

                elif action_type == 'change_status':
                    transition_id = action_data.get('transition_id')
                    success, error = self.jira_client.transition_issue(issue_key, transition_id)
                    if success:
                        await self.send_message(chat_id, f"✅ Status of {issue_key} changed successfully.")
                    else:
                        await self.send_message(chat_id, f"❌ Failed to change status: {error}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in callback_data: {data}")
                await self.send_message(chat_id, "❌ Invalid action data. Please try again or contact support.")
            except Exception as e:
                logger.error(f"Error confirming action: {e}")
                await self.send_message(chat_id, "❌ An unexpected error occurred while confirming the action.")

        elif data.startswith("transition:"):
            try:
                parts = data.split(":")
                if len(parts) == 3:
                    _, issue_key, transition_id = parts
                    action_data = {
                        'action': 'change_status',
                        'issue_key': issue_key,
                        'transition_id': transition_id
                    }
                    confirmation_message = f"Confirm: Change status of *{issue_key}*?"
                    keyboard = [
                        [InlineKeyboardButton("Yes", callback_data=f"confirm_action:{json.dumps(action_data)}")],
                        [InlineKeyboardButton("No", callback_data="cancel_action")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await self.send_message(chat_id, confirmation_message, reply_markup=reply_markup)
                else:
                    await self.send_message(chat_id, "❌ Invalid transition data.")
            except Exception as e:
                logger.error(f"Error handling transition callback: {e}")
                await self.send_message(chat_id, "❌ An error occurred while processing status change options.")

        elif data == "cancel_action":
            await self.send_message(chat_id, "Action cancelled.")

    async def handle_settings_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callbacks from the settings inline keyboard."""
        query = update.callback_query
        await query.answer()
        chat_id = query.message.chat_id
        user_id = update.effective_user.id

        data = query.data
        if data.startswith("toggle_setting:"):
            setting_name = data.replace("toggle_setting:", "")
            current_state = self._notification_preferences.get(user_id, {}).get(setting_name, True)
            new_state = not current_state

            if user_id not in self._notification_preferences:
                self._notification_preferences[user_id] = {}
            self._notification_preferences[user_id][setting_name] = new_state

            await self.settings_command(update, context) # Refresh settings message

        elif data == "main_menu":
            # You can send a main menu message or just dismiss the settings message
            await self.send_message(chat_id, "Returning to main menu.")

    async def handle_natural_language_query(self, query: str, user_id: int):
        """Handles natural language queries that are not directly mapped to Jira actions."""
        try:
            # Add user's query to conversation history
            self.conversation_history[user_id].append({"role": "user", "content": query})

            response_text = self.openai_client.handle_natural_language_query_ai(self.conversation_history[user_id], query)
            # Add AI's response to conversation history
            self.conversation_history[user_id].append({"role": "assistant", "content": response_text})
            return response_text
        except Exception as e:
            logger.error(f"Error in handle_natural_language_query: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."

    async def generate_smart_summary(self, issue_key: str, summary_type: str = 'executive') -> str:
        """Generate a smart summary for a ticket based on the provided issue_key and summary_type."""
        issue_data = self.get_jira_issue(issue_key)
        if not issue_data:
            return f"❌ Could not find ticket {issue_key}"

        summary = issue_data['fields'].get('summary', '')
        description = issue_data['fields'].get('description', '')
        status = issue_data['fields']['status']['name']
        priority = issue_data['fields']['priority']['name']
        assignee = issue_data['fields']['assignee']['displayName'] if issue_data['fields']['assignee'] else 'Unassigned'
        assignee_email = issue_data['fields']['assignee']['emailAddress'] if issue_data['fields']['assignee'] and 'emailAddress' in issue_data['fields']['assignee'] else ''
        jira_url = f"https://{self.jira_client.jira_domain}/browse/{issue_key}"

        if summary_type == 'executive':
            return f"""# Ticket Analysis for [{issue_key}]({jira_url})

## Overview
| Field    | Value |
|------------|---------|
| Title    | {summary} |
| Status   | {status} |
| Priority | {priority} |
| Assignee | [{assignee}](mailto:{assignee_email}) |

## Smart Summary
{self.openai_client.generate_smart_summary(summary, description)}
"""
        elif summary_type == 'timeline':
            return f"""# Timeline Summary for [{issue_key}]({jira_url})

## Key Dates
- **Opened:** {datetime.fromisoformat(issue_data['fields']['created']).strftime('%Y-%m-%d %H:%M:%S')}
- **Last Updated:** {datetime.fromisoformat(issue_data['fields']['updated']).strftime('%Y-%m-%d %H:%M:%S')}

## Description
{description}
"""
        else:
            return f"❌ Invalid summary type: {summary_type}"

    def get_comments(self, issue_key):
        """Fetches comments for a Jira issue (via JiraClient)."""
        comments, error = self.jira_client.get_comments(issue_key)
        if error:
            logger.error(f"Error getting comments for {issue_key}: {error}")
            return None
        return comments

    async def handle_message(self, message: str, chat_id: int, user_id: int) -> None:
        """Handle incoming messages"""
        try:
            # Initialize thread context for user if not exists
            if user_id not in self._thread_context:
                self._thread_context[user_id] = {'last_ticket': None, 'goals': [], 'context': {}}

            # Check if user is in pending comment state
            if user_id in self._pending_comment:
                issue_key = self._pending_comment[user_id]
                # Add the comment
                success, error = self.jira_client.add_comment_to_issue(issue_key, message)
                if success:
                    await self.send_message(chat_id, f"✅ Comment added to {issue_key}:\n\n{message}")
                else:
                    await self.send_message(chat_id, f"❌ Failed to add comment: {error}")
                # Clear pending state
                del self._pending_comment[user_id]
                return

            # First try to extract intent using AI
            entities = self.openai_client.extract_action_entities(message)
            logger.info(f"Extracted entities: {entities}")
            
            # Check for dynamic intent chaining
            if not entities.get('issue_key') and user_id in self._last_ticket_context:
                # If no issue_key in current message, use the last context
                entities['issue_key'] = self._last_ticket_context[user_id]
                logger.info(f"Chaining intent with last ticket context: {entities['issue_key']}")

            # Update thread context with the last ticket
            if entities.get('issue_key'):
                self._thread_context[user_id]['last_ticket'] = entities['issue_key']

            # Handle create ticket requests
            if entities.get('action') == 'create' or 'create a jira' in message.lower():
                project_key = entities.get('project_key')
                summary = entities.get('summary', '')
                description = entities.get('description', '')
                issue_type = entities.get('issue_type', 'Task')

                if not project_key:
                    available_projects = await self.fetch_available_projects()
                    if available_projects:
                        await self.send_message(chat_id, f"Please provide a project key from the following list: {', '.join(available_projects)}")
                    else:
                        await self.send_message(chat_id, "No projects available. Please check your Jira configuration.")
                    return

                if not summary:
                    await self.send_message(chat_id, "Please provide a summary for the ticket.")
                    return

                # Confirm ticket creation
                confirmation_message = f"Creating a ticket in project {project_key} with summary: {summary}. Confirm? (yes/no)"
                await self.send_message(chat_id, confirmation_message)
                # Wait for user confirmation
                # This is a placeholder for actual confirmation logic
                # You may need to implement a way to handle user responses

                success, error = self.create_jira_ticket(project_key, summary, description, issue_type)
                if success:
                    await self.send_message(chat_id, f"✅ Ticket created successfully in project {project_key}!")
                else:
                    await self.send_message(chat_id, f"❌ Failed to create ticket: {error}")
                return

            # Handle analyze action
            if entities.get('action') == 'analyze':
                issue_key = entities.get('issue_key')
                if not issue_key:
                    await self.send_message(chat_id, "❌ I couldn't find a valid Jira ticket number in your message. Please include a ticket number like CBT-6030.")
                    return

                # Store the issue_key for future context
                self._last_ticket_context[user_id] = issue_key

                # Use the analyze_ticket functionality
                issue_data = self.get_jira_issue(issue_key)
                if not issue_data:
                    await self.send_message(chat_id, f"❌ Could not find ticket {issue_key}")
                    return

                # Get ticket details
                summary = issue_data['fields'].get('summary', '')
                description = issue_data['fields'].get('description', '')
                status = issue_data['fields']['status']['name']
                priority = issue_data['fields']['priority']['name']
                assignee = issue_data['fields']['assignee']['displayName'] if issue_data['fields']['assignee'] else 'Unassigned'
                assignee_email = issue_data['fields']['assignee']['emailAddress'] if issue_data['fields']['assignee'] and 'emailAddress' in issue_data['fields']['assignee'] else ''
                jira_url = f"https://{self.jira_client.jira_domain}/browse/{issue_key}"

                # Generate smart summary
                summary_analysis = self.openai_client.generate_smart_summary(summary, description)

                # Format response with Markdown table and lists
                response = f"""🔍 *Ticket Analysis for [{issue_key}]({jira_url})*

## Overview
| Field    | Value |
|------------|---------|
| *Title*    | {summary} |
| *Status*   | {status} |
| *Priority* | {priority} |
| *Assignee* | [{assignee}](mailto:{assignee_email}) |

## Smart Summary
{summary_analysis}
"""

                # Add inline keyboard buttons for quick actions
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Close Ticket", callback_data=f"close_ticket:{issue_key}"),
                        InlineKeyboardButton("👤 Assign to Me", callback_data=f"assign_to_me:{issue_key}"),
                    ],
                    [
                        InlineKeyboardButton("💬 Add Comment", callback_data=f"add_comment:{issue_key}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await self.send_message(chat_id, response, reply_markup=reply_markup)
                return

            # Handle other actions as before
            if entities and entities.get('action') == 'add_comment':
                issue_key = entities.get('issue_key')
                comment_text = entities.get('comment_text', '')
                
                if not issue_key:
                    await self.send_message(chat_id, "❌ I couldn't find a valid Jira ticket number in your message. Please include a ticket number like CBT-6030.")
                    return
                    
                if not comment_text:
                    await self.send_message(chat_id, "❌ I couldn't find the comment text. Please provide what you want to add as a comment.")
                    return
                
                # Add the comment
                success, error = self.jira_client.add_comment_to_issue(issue_key, comment_text)
                if success:
                    await self.send_message(chat_id, f"✅ Comment added to {issue_key}:\n\n{comment_text}")
                else:
                    await self.send_message(chat_id, f"❌ Failed to add comment: {error}")
                return
                
            # Handle natural dialog for status transitions
            if entities.get('action') == 'change_status':
                issue_key = entities.get('issue_key')
                if not issue_key:
                    await self.send_message(chat_id, "❌ I couldn't find a valid Jira ticket number in your message. Please include a ticket number like CBT-6030.")
                    return
                    
                # Store the issue_key for future context
                self._last_ticket_context[user_id] = issue_key

                # Map human phrasing to Jira status
                target_status = entities.get('status', '').lower()
                mapped_status = self._status_mapping.get(target_status, target_status)

                # Get available transitions
                transitions, error = self.jira_client.get_available_transitions(issue_key)
                if error or not transitions:
                    await self.send_message(chat_id, f"❌ Could not fetch transitions for {issue_key}: {error or 'No transitions found.'}")
                    return
                    
                # Find the matching transition
                matching_transition = next((t for t in transitions if t['name'].lower() == mapped_status.lower()), None)
                if not matching_transition:
                    await self.send_message(chat_id, f"❌ No matching transition found for status '{mapped_status}'.")
                    return
                    
                # Ask for confirmation
                confirmation_message = f"Got it. Changing {issue_key} to {mapped_status}. Confirm?"
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Yes", callback_data=f"confirm_transition:{issue_key}:{matching_transition['id']}"),
                        InlineKeyboardButton("❌ No", callback_data="cancel_action")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await self.send_message(chat_id, confirmation_message, reply_markup=reply_markup)
                return
                
            # Handle smart summary request
            if entities.get('action') == 'summarize':
                issue_key = entities.get('issue_key')
                summary_type = entities.get('summary_type', 'executive')
                if not issue_key:
                    await self.send_message(chat_id, "❌ I couldn't find a valid Jira ticket number in your message. Please include a ticket number like CBT-6030.")
                    return
                # Generate smart summary
                summary = await self.generate_smart_summary(issue_key, summary_type)
                await self.send_message(chat_id, summary)
                return
                    
            # If no specific action was identified, try JQL search
            jql = self.openai_client.convert_nl_to_jql(message)
            if jql == "INVALID_JQL":
                await self.send_message(
                    chat_id,
                    "❌ I couldn't understand that as a valid Jira query. Could you please rephrase or be more specific?"
                )
                return
                
            issues, error = self.jira_client.search_issues_by_jql(jql)
            if error:
                await self.send_message(chat_id, f"❌ Error searching Jira: {error}")
                return
                
            if not issues:
                await self.send_message(chat_id, "ℹ️ No matching issues found")
                return
                
            # Format and send results
            response = "🔍 *Search Results:*\n\n"
            for issue in issues:
                response += f"📋 {issue['key']}: {issue['fields']['summary']}\n"
                response += f"👤 Assignee: {issue['fields']['assignee']['displayName'] if issue['fields']['assignee'] else 'Unassigned'}\n"
                response += f"📊 Status: {issue['fields']['status']['name']}\n"
                response += f"⭐ Priority: {issue['fields']['priority']['name']}\n\n"
            await self.send_message(chat_id, response)
            
            # Handle list projects command
            if message.lower() == 'list projects':
                await self.list_projects(message)
                return
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_message(chat_id, "❌ An error occurred while processing your request")

    async def handle_callback_queries(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Routes all callback queries to appropriate handler."""
        query = update.callback_query
        await query.answer()
        data = query.data
        chat_id = query.message.chat_id
        user_id = update.effective_user.id

        if data.startswith("toggle_setting:"):
            await self.handle_settings_callback(update, context)
        elif data.startswith("confirm_action:") or data.startswith("transition:") or data == "cancel_action":
            await self.handle_action_confirmation(update, context)
        elif data == "main_menu":
            await self.send_message(query.message.chat_id, "Returning to main menu.")
        elif data.startswith("close_ticket:"):
            # Close the ticket (transition to Done/Closed)
            issue_key = data.split(":")[1]
            # Find the correct transition for closing
            transitions, error = self.jira_client.get_available_transitions(issue_key)
            if error or not transitions:
                await self.send_message(chat_id, f"❌ Could not fetch transitions for {issue_key}: {error or 'No transitions found.'}")
                return
            # Try to find a 'Done' or 'Closed' transition
            close_transition = next((t for t in transitions if t['name'].lower() in ['done', 'closed']), None)
            if not close_transition:
                await self.send_message(chat_id, f"❌ No 'Done' or 'Closed' transition available for {issue_key}.")
                return
            success, error = self.jira_client.change_issue_status(issue_key, close_transition['name'])
            if success:
                await self.send_message(chat_id, f"✅ Ticket {issue_key} has been closed.")
            else:
                await self.send_message(chat_id, f"❌ Failed to close ticket {issue_key}: {error}")
        elif data.startswith("assign_to_me:"):
            # Assign the ticket to the current user
            issue_key = data.split(":")[1]
            # Get user's Jira email/accountId
            jira_email = self._user_data.get(user_id, {}).get('jira_email', None)
            if not jira_email:
                await self.send_message(chat_id, "❌ Your Jira email is not set up. Please ensure JIRA_EMAIL is configured in .env.")
                return
            account_id, error = self.jira_client.get_user_account_id(jira_email)
            if not account_id:
                await self.send_message(chat_id, f"❌ Could not find your Jira account: {error}")
                return
            success, error = self.jira_client.assign_issue(issue_key, account_id)
            if success:
                await self.send_message(chat_id, f"✅ Ticket {issue_key} assigned to you.")
            else:
                await self.send_message(chat_id, f"❌ Failed to assign ticket: {error}")
        elif data.startswith("add_comment:"):
            # Prompt the user to reply with a comment
            issue_key = data.split(":")[1]
            self._pending_comment[user_id] = issue_key
            await self.send_message(chat_id, f"💬 Please reply with your comment for {issue_key}.")
        elif data == "confirm_create_issues":
            plan = self._thread_context[user_id].get('pending_plan')
            if not plan:
                await self.send_message(chat_id, "❌ No plan found to create issues.")
                return
            project_key = self._thread_context[user_id].get('pending_project_key')
            if not project_key:
                await self.send_message(chat_id, "Please provide the Jira project key (e.g., DEV, ABC):")
                self._thread_context[user_id]['awaiting_project_key'] = True
                return
            results = self.create_jira_issues_from_plan(plan, project_key)
            await self.send_message(chat_id, f"✅ Created issues:\n{results}")
            self._thread_context[user_id]['pending_plan'] = None
            self._thread_context[user_id]['pending_project_key'] = None
        else:
            await self.send_message(query.message.chat_id, "Unknown action.")

    async def analyze_ticket(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze a ticket using AI capabilities."""
        try:
            # Extract ticket key from command arguments
            if not context.args:
                await self.send_message(update.effective_chat.id, "❌ Please provide a ticket key. Usage: /analyze_ticket PROJ-123")
                return

            ticket_key = context.args[0].upper()
            issue_data = self.get_jira_issue(ticket_key)
            print(f"issue_data type: {type(issue_data)}, value: {issue_data}")
            logger.debug(f"issue_data type: {type(issue_data)}, value: {issue_data}")
            
            if not issue_data:
                await self.send_message(update.effective_chat.id, f"❌ Could not find ticket {ticket_key}")
                return

            # Get ticket details
            summary = issue_data['fields'].get('summary', '')
            description = issue_data['fields'].get('description', '')
            comments, _ = self.jira_client.get_comments(ticket_key)

            # Generate comprehensive analysis
            analysis_message = "🔍 *Ticket Analysis*\n\n"
            
            # Smart summary
            summary_analysis = self.openai_client.generate_smart_summary(summary, description)
            analysis_message += f"*Smart Summary:*\n{summary_analysis}\n\n"
            
            # Priority prediction
            priority_analysis = self.openai_client.predict_ticket_priority(summary, description, comments)
            analysis_message += f"*Priority Analysis:*\n{priority_analysis['analysis']}\n\n"
            
            # Duplicate detection
            similar_tickets_result = self.jira_client.search_issues_by_jql(f"project = {ticket_key.split('-')[0]} AND key != {ticket_key}")
            logger.debug(f"similar_tickets_result type: {type(similar_tickets_result)}, value: {similar_tickets_result}")
            if similar_tickets_result and isinstance(similar_tickets_result, tuple) and similar_tickets_result[0]:
                similar_tickets = similar_tickets_result[0]
                duplicate_analysis = self.openai_client.detect_duplicate_tickets(issue_data, similar_tickets)
                analysis_message += f"*Duplicate Analysis:*\n{duplicate_analysis['analysis']}\n\n"
            
            # Resolution time prediction
            historical_data_result = self.jira_client.search_issues_by_jql(
                f"project = {ticket_key.split('-')[0]} AND status = Done ORDER BY updated DESC"
            )
            logger.debug(f"historical_data_result type: {type(historical_data_result)}, value: {historical_data_result}")
            if historical_data_result and isinstance(historical_data_result, tuple) and historical_data_result[0]:
                historical_data = historical_data_result[0][:10]  # Get last 10 resolved tickets
                resolution_prediction = self.openai_client.predict_resolution_time(issue_data, historical_data)
                analysis_message += f"*Resolution Prediction:*\n{resolution_prediction['prediction']}\n\n"

            await self.send_message(update.effective_chat.id, analysis_message)

        except Exception as e:
            logger.error(f"Error in analyze_ticket: {e}")
            await self.send_message(update.effective_chat.id, f"❌ An error occurred while analyzing the ticket: {e}")

    async def analyze_workload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analyze team workload distribution."""
        try:
            # Get team members and their current tickets
            team_members = []
            tickets = []
            
            # Fetch team members and their tickets
            jql_query = "statusCategory != Done"
            all_tickets, error = self.jira_client.search_issues_by_jql(jql_query)
            
            if error:
                await self.send_message(update.effective_chat.id, f"❌ Error fetching tickets: {error}")
                return

            # Process tickets and build team member workload
            for ticket in all_tickets:
                assignee = ticket['fields'].get('assignee', {})
                if assignee:
                    assignee_name = assignee.get('displayName', 'Unassigned')
                    if assignee_name not in [m['name'] for m in team_members]:
                        team_members.append({
                            'name': assignee_name,
                            'workload': 1,
                            'expertise': self._get_user_expertise(assignee_name)
                        })
                    else:
                        for member in team_members:
                            if member['name'] == assignee_name:
                                member['workload'] += 1
                tickets.append({
                    'key': ticket['key'],
                    'priority': ticket['fields']['priority']['name'],
                    'status': ticket['fields']['status']['name']
                })

            # Get AI analysis
            workload_analysis = self.openai_client.analyze_workload_distribution(team_members, tickets)
            
            # Format and send response
            response = "📊 *Team Workload Analysis*\n\n"
            response += workload_analysis['analysis']
            
            await self.send_message(update.effective_chat.id, response)

        except Exception as e:
            logger.error(f"Error in analyze_workload: {e}")
            await self.send_message(update.effective_chat.id, "❌ An error occurred while analyzing workload.")

    async def auto_assign_ticket(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Automatically assign a ticket based on AI analysis."""
        try:
            if not context.args:
                await self.send_message(update.effective_chat.id, "❌ Please provide a ticket key. Usage: /auto_assign PROJ-123")
                return

            ticket_key = context.args[0].upper()
            issue_data = self.get_jira_issue(ticket_key)
            
            if not issue_data:
                await self.send_message(update.effective_chat.id, f"❌ Could not find ticket {ticket_key}")
                return

            # Get team members
            team_members = self._get_team_members()
            
            # Get AI suggestion
            suggestion = self.openai_client.suggest_assignee(issue_data, team_members)
            
            if 'error' in suggestion:
                await self.send_message(update.effective_chat.id, f"❌ {suggestion['error']}")
                return

            # Format and send response
            response = f"🤖 *Auto-Assignment Suggestion for {ticket_key}*\n\n"
            response += suggestion['suggestion']
            
            # Add confirmation buttons
            keyboard = [
                [
                    InlineKeyboardButton("✅ Confirm", callback_data=f"confirm_assign:{ticket_key}"),
                    InlineKeyboardButton("❌ Cancel", callback_data="cancel_action")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.send_message(update.effective_chat.id, response, reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in auto_assign_ticket: {e}")
            await self.send_message(update.effective_chat.id, "❌ An error occurred while suggesting assignment.")

    async def predict_ticket_metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Predict various metrics for a ticket."""
        try:
            if not context.args:
                await self.send_message(update.effective_chat.id, "❌ Please provide a ticket key. Usage: /predict PROJ-123")
                return

            ticket_key = context.args[0].upper()
            issue_data = self.get_jira_issue(ticket_key)
            
            if not issue_data:
                await self.send_message(update.effective_chat.id, f"❌ Could not find ticket {ticket_key}")
                return

            # Get historical data
            historical_data = self.jira_client.search_issues_by_jql(
                f"project = {ticket_key.split('-')[0]} AND status = Done ORDER BY updated DESC"
            )[0][:10]  # Get last 10 resolved tickets

            # Get predictions
            resolution_prediction = self.openai_client.predict_resolution_time(issue_data, historical_data)
            
            # Format and send response
            response = f"🔮 *Predictions for {ticket_key}*\n\n"
            response += f"*Resolution Time Prediction:*\n{resolution_prediction['prediction']}\n\n"
            
            await self.send_message(update.effective_chat.id, response)

        except Exception as e:
            logger.error(f"Error in predict_ticket_metrics: {e}")
            await self.send_message(update.effective_chat.id, "❌ An error occurred while generating predictions.")

    def _get_user_expertise(self, username: str) -> str:
        """Get user expertise based on their ticket history."""
        try:
            # Search for user's resolved tickets
            jql_query = f"assignee = \"{username}\" AND status = Done ORDER BY updated DESC"
            resolved_tickets, _ = self.jira_client.search_issues_by_jql(jql_query)
            
            if not resolved_tickets:
                return "General"
            
            # Analyze ticket types and components
            ticket_types = {}
            for ticket in resolved_tickets[:50]:  # Look at last 50 tickets
                issue_type = ticket['fields']['issuetype']['name']
                ticket_types[issue_type] = ticket_types.get(issue_type, 0) + 1
            
            # Get most common ticket type
            most_common = max(ticket_types.items(), key=lambda x: x[1])[0]
            return most_common
            
        except Exception as e:
            logger.error(f"Error getting user expertise: {e}")
            return "General"

    def _get_team_members(self) -> list:
        """Get list of team members with their current workload."""
        try:
            # Get all active team members
            jql_query = "statusCategory != Done"
            all_tickets, _ = self.jira_client.search_issues_by_jql(jql_query)
            
            team_members = {}
            for ticket in all_tickets:
                assignee = ticket['fields'].get('assignee', {})
                if assignee:
                    name = assignee.get('displayName', 'Unassigned')
                    if name not in team_members:
                        team_members[name] = {
                            'name': name,
                            'workload': 1,
                            'expertise': self._get_user_expertise(name)
                        }
                    else:
                        team_members[name]['workload'] += 1
            
            return list(team_members.values())
            
        except Exception as e:
            logger.error(f"Error getting team members: {e}")
            return []

    async def handle_action_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle confirmation of automated actions."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        if data.startswith("confirm_assign:"):
            ticket_key = data.replace("confirm_assign:", "")
            # Get the suggested assignee from the message
            message_text = query.message.text
            assignee_name = self._extract_assignee_from_message(message_text)
            
            if assignee_name:
                # Update the ticket
                success, error = self.jira_client.update_issue_assignee(ticket_key, assignee_name)
                if success:
                    # Generate and add auto comment
                    issue_data = self.get_jira_issue(ticket_key)
                    comment = self.openai_client.generate_auto_comment(
                        issue_data,
                        f"Automatically assigned to {assignee_name} based on AI analysis"
                    )
                    self.jira_client.add_comment_to_issue(ticket_key, comment)
                    
                    await self.send_message(
                        query.message.chat_id,
                        f"✅ Successfully assigned {ticket_key} to {assignee_name}"
                    )
                else:
                    await self.send_message(
                        query.message.chat_id,
                        f"❌ Failed to assign ticket: {error}"
                    )
            else:
                await self.send_message(
                    query.message.chat_id,
                    "❌ Could not determine the suggested assignee"
                )
        
        elif data == "cancel_action":
            await self.send_message(
                query.message.chat_id,
                "❌ Action cancelled"
            )

    def _extract_assignee_from_message(self, message: str) -> str:
        """Extract assignee name from the AI suggestion message."""
        try:
            # Look for the recommended assignee in the message
            lines = message.split('\n')
            for line in lines:
                if "Recommended assignee:" in line:
                    return line.split("Recommended assignee:")[1].strip()
            return None
        except Exception as e:
            logger.error(f"Error extracting assignee from message: {e}")
            return None

    async def generate_smart_summary(self, issue_key: str, summary_type: str = 'executive') -> str:
        """Generate a smart summary for a ticket."""
        issue_data = self.get_jira_issue(issue_key)
        if not issue_data:
            return "❌ Could not find ticket details."

        summary = issue_data['fields'].get('summary', '')
        description = issue_data['fields'].get('description', '')
        status = issue_data['fields']['status']['name']
        priority = issue_data['fields']['priority']['name']
        assignee = issue_data['fields']['assignee']['displayName'] if issue_data['fields']['assignee'] else 'Unassigned'
        created = issue_data['fields']['created']
        updated = issue_data['fields']['updated']

        if summary_type == 'executive':
            # Generate a 2-line executive summary
            executive_summary = f"📋 {summary}\n👤 Assigned to: {assignee}, Status: {status}, Priority: {priority}"
            return executive_summary
        elif summary_type == 'timeline':
            # Generate a timeline summary
            timeline_summary = f"Opened on {created}\nLast updated on {updated}"
            return timeline_summary
        else:
            return "❌ Invalid summary type requested."

    async def create_jira_ticket(self, project_key: str, summary: str, description: str, issue_type: str = 'Task') -> tuple:
        """Create a new Jira ticket under the specified project."""
        try:
            # Call the Jira client to create a ticket
            success, error = self.jira_client.create_issue(project_key, summary, description, issue_type)
            return success, error
        except Exception as e:
            logger.error(f"Error creating Jira ticket: {e}")
            return False, str(e)

    async def fetch_available_projects(self):
        """Fetch available projects from Jira."""
        try:
            self.logger.info("Fetching available projects from Jira...")
            projects = self.jira_client.projects()
            project_keys = [project.key for project in projects]
            self.logger.info(f"Fetched projects: {project_keys}")
            return project_keys
        except Exception as e:
            self.logger.error(f"Error fetching projects: {e}")
            return []

    async def list_projects(self, message):
        """List all available projects from Jira."""
        available_projects = await self.fetch_available_projects()
        if available_projects:
            await message.reply_text(f"Available projects: {', '.join(available_projects)}")
        else:
            await message.reply_text("No projects available. Please check your Jira configuration.")

    async def auto_create_issues(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Initiate the PRD/requirements upload process."""
        await self.send_message(
            update.effective_chat.id,
            "📄 Please upload your PRD/requirements document (TXT, PDF, DOCX, PDF) or paste the requirements as a message."
        )
        user_id = update.effective_user.id
        if user_id not in self._thread_context:
            self._thread_context[user_id] = {}
        self._thread_context[user_id]['awaiting_prd'] = True

    async def handle_prd_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded PRD files."""
        user_id = update.effective_user.id
        if not self._thread_context.get(user_id, {}).get('awaiting_prd'):
            return  # Ignore if not expecting PRD
        document = update.message.document
        file = await context.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{document.file_name}") as tmp_file:
            await file.download_to_drive(tmp_file.name)
            prd_text = self.extract_text_from_file(tmp_file.name)
        os.remove(tmp_file.name)
        await self.process_prd_text(update, context, prd_text)

    async def process_prd_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE, prd_text: str):
        """Process PRD text, analyze with AI, and show preview."""
        user_id = update.effective_user.id
        self._thread_context[user_id]['awaiting_prd'] = False
        await self.send_message(update.effective_chat.id, "🤖 Analyzing requirements with AI...")
        plan = self.analyze_prd_with_ai(prd_text)
        self._thread_context[user_id]['pending_plan'] = plan
        preview = self.format_plan_preview(plan)
        keyboard = [
            [InlineKeyboardButton("✅ Confirm", callback_data="confirm_create_issues")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_action")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await self.send_message(update.effective_chat.id, f"Here's the suggested breakdown:\n\n{preview}", reply_markup=reply_markup)

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from TXT, PDF, or DOCX files."""
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.endswith('.pdf'):
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(page.extract_text() or '' for page in reader.pages)
            except Exception as e:
                logger.error(f"PDF extraction error: {e}")
                return ""
        elif file_path.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                logger.error(f"DOCX extraction error: {e}")
                return ""
        else:
            return ""

    def analyze_prd_with_ai(self, prd_text: str) -> dict:
        """Use OpenAI to break down PRD into Tasks and Sub-tasks."""
        prompt = (
            "You are a Jira expert. Given the following product requirements, break them down into a hierarchy of Tasks and Sub-tasks. "
            "Return the result as a JSON object with this structure:\n"
            "{\"tasks\": [{\"title\": ..., \"description\": ..., \"subtasks\": [{\"title\": ..., \"description\": ...}]}]}\n"
            f"Requirements:\n{prd_text}"
        )
        response = self.openai_client.complete(prompt, max_tokens=1500)
        try:
            plan = json.loads(response)
        except Exception as e:
            logger.error(f"AI plan JSON decode error: {e}, response: {response}")
            plan = {}
        return plan

    def format_plan_preview(self, plan: dict) -> str:
        """Format the AI plan for user preview (Tasks and Sub-tasks)."""
        if not plan or 'tasks' not in plan:
            return "❌ Could not generate a breakdown."
        lines = []
        for task in plan['tasks']:
            lines.append(f"*Task:* {task['title']}\n  _{task.get('description', '')}_")
            for sub in task.get('subtasks', []):
                lines.append(f"  - Subtask: {sub['title']}\n    _{sub.get('description', '')}_")
        return "\n".join(lines)

    def create_jira_issues_from_plan(self, plan: dict, project_key: str) -> str:
        """Create Stories and Sub-tasks in Jira from the AI plan."""
        results = []
        for task in plan.get('tasks', []):
            task_key, error = self.jira_client.create_issue(
                project_key=project_key,
                summary=task['title'],
                description=task.get('description', ''),
                issue_type="Story"  # Use 'Story' for parent
            )
            if not task_key:
                results.append(f"Failed to create Story: {task['title']} ({error})")
                continue
            results.append(f"Story: {task_key} - {task['title']}")
            for sub in task.get('subtasks', []):
                sub_key, error = self.jira_client.create_issue(
                    project_key=project_key,
                    summary=sub['title'],
                    description=sub.get('description', ''),
                    issue_type="Subtask",
                    parent=task_key  # Pass parent Story key
                )
                if not sub_key:
                    results.append(f"  Failed to create Subtask: {sub['title']} ({error})")
                else:
                    results.append(f"  Subtask: {sub_key} - {sub['title']}")
        return "\n".join(results)

def main():
    # Check for required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'JIRA_URL', 'JIRA_EMAIL', 'JIRA_API_TOKEN', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("Please create a .env file in the root directory with the following:")
        print("TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN\nJIRA_URL=YOUR_JIRA_URL\nJIRA_EMAIL=YOUR_JIRA_EMAIL\nJIRA_API_TOKEN=YOUR_JIRA_API_TOKEN\nOPENAI_API_KEY=YOUR_OPENAI_API_KEY")
        return

    try:
        bot = JiraTrackerBot()
        
        # Register the error handler
        bot.application.add_error_handler(error_handler)

        # Graceful shutdown
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: loop.stop())
            loop.add_signal_handler(signal.SIGTERM, lambda: loop.stop())
        except NotImplementedError: # For Windows, signal handlers might not be implemented
            pass

        logger.info("Starting Jira Tracker Bot...")
        bot.application.run_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Jira Tracker Bot stopped.")
    except Exception as e:
        logger.error(f"Error initializing bot: {e}")
        print(f"Error initializing bot: {e}")

def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a message to the user."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if update and update.effective_chat:
        if isinstance(context.error, telegram.error.Conflict):
            error_message = "Another bot instance is running. Please stop the other instance and try again."
        else:
            error_message = "An unexpected error occurred. Please try again later."
        asyncio.create_task(context.bot.send_message(chat_id=update.effective_chat.id, text=error_message))

if __name__ == '__main__':
    main()