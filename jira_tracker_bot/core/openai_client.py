import os
import logging
from openai import OpenAI, APIError
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = "gpt-4-turbo" # Default model

    def _call_openai(self, prompt: str, max_tokens: int, temperature: float, stop: list = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise

    def extract_issue_key_ai(self, text: str):
        """Extract Jira issue key from text using AI."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Extract Jira issue key from the text. Return only the key (e.g., PROJ-123) or None if not found."},
                    {"role": "user", "content": text}
                ]
            )
            extracted_key = response.choices[0].message.content.strip()
            if extracted_key and len(extracted_key) <= 20 and all(c.isalnum() or c == '-' for c in extracted_key):
                return extracted_key.upper()
            else:
                return None
        except APIError as e:
            logger.error(f"OpenAI API Error in extract_issue_key_ai: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in extract_issue_key_ai: {e}")
            return None

    def generate_smart_summary(self, summary: str, description: str) -> str:
        """Generate a smart summary of the issue using AI."""
        prompt = f"""Analyze this Jira ticket and provide a concise summary highlighting key points:
        Summary: {summary}
        Description: {description}
        
        Please provide:
        1. Main objective
        2. Key requirements
        3. Technical considerations
        4. Potential risks
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating smart summary: {e}")
            return "Unable to generate smart summary at this time."

    def analyze_comments_ai(self, comment_texts: list[str]):
        """Analyze comments using AI to extract key points."""
        if not comment_texts:
            return "No comments to analyze."

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Analyze these comments and extract key points and action items."},
                    {"role": "user", "content": "\n".join(comment_texts)}
                ]
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"OpenAI API Error in analyze_comments_ai: {e}")
            return "Unable to analyze comments due to API error."
        except Exception as e:
            logger.error(f"Unexpected error in analyze_comments_ai: {e}")
            return "Unable to analyze comments."

    def generate_user_friendly_error(self, error: str):
        """Generate user-friendly error messages using AI. THIS IS NOW BACK TO AI GENERATION."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Convert technical error messages into user-friendly explanations."},
                    {"role": "user", "content": error}
                ]
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"Error generating AI error message from OpenAI API: {e}")
            return "An AI error occurred. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error generating AI error message: {e}")
            return "An error occurred. Please try again."

    def handle_natural_language_query_ai(self, messages: list[dict], query: str):
        """Handle natural language queries using AI."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a Jira assistant. Help users find and understand Jira tickets."},
                    *messages,
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content.strip()
        except APIError as e:
            logger.error(f"OpenAI API Error in handle_natural_language_query_ai: {e}")
            return f"I'm having trouble understanding your query due to an API error. Raw error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in handle_natural_language_query_ai: {e}")
            return f"I'm having trouble understanding your query. Raw error: {e}"

    def parse_jira_action(self, query: str):
        """Parses a natural language query to identify a Jira action and its parameters."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the user's request and extract ONLY explicit Jira *manipulation* actions (e.g., 'change status', 'add comment', 'assign') and their parameters (issue key, new status, comment text, assignee). If an action is found, return a JSON object like: {\'action\': \'change_status\', \'issue_key\': \'PROJ-123\', \'status\': \'Done\'} or {\'action\': \'add_comment\', \'issue_key\': \'PROJ-123\', \'comment\': \'Your comment here.\'}. If the request is for information (e.g., 'what's the status', 'find tickets') or does not contain a clear manipulation action, return {\'action\': None}. Be precise and only return JSON."},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            action_json_str = response.choices[0].message.content.strip()
            action_data = json.loads(action_json_str)
            return action_data
        except APIError as e:
            logger.error(f"OpenAI API Error in parse_jira_action: {e}")
            return {'action': None, 'error': f"AI parsing error: {e}"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error in parse_jira_action: {e}. Raw response: {action_json_str}")
            return {'action': None, 'error': "Could not parse AI response."}
        except Exception as e:
            logger.error(f"Unexpected error in parse_jira_action: {e}")
            return {'action': None, 'error': f"An unexpected error occurred during action parsing: {e}"}

    def convert_nl_to_jql(self, query: str, user_email: str = None):
        """Converts a natural language query into a JQL string using AI."""
        system_message = (
            "You are an expert in Jira Query Language (JQL). Your ONLY task is to convert natural language queries into valid JQL strings. "
            "Return ONLY the JQL string. Do NOT include any conversational text, explanations, or markdown formatting (e.g., ```jql```, ```json```). "
            "Focus on issue attributes like 'status', 'assignee', 'reporter', 'priority', 'project', 'type', 'summary', 'description', 'created', 'updated'. "
            "Always ensure the JQL is syntactically correct and can be directly used in Jira. "
            "If a user's email is provided, use 'currentUser()' if the query implies their own tickets (e.g., 'my tickets', 'tickets assigned to me'), otherwise use the exact email in quotes. "
            "If the query cannot be accurately converted to JQL, return the exact string 'INVALID_JQL'."
        )
        user_prompt = f"Convert the following query to JQL: '{query}'."
        if user_email:
            user_prompt += f" Assume my email is {user_email}."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]
            )
            jql = response.choices[0].message.content.strip()
            
            # Aggressive post-processing to remove any extra text/markdown
            if jql.lower().startswith("```jql"): # Remove markdown code block if present
                jql = jql[len("```jql"):].strip()
            if jql.lower().endswith("```"): # Remove closing markdown code block
                jql = jql[:-len("```")].strip()
            
            # Remove any leading/trailing conversational text that might remain
            if "jql:" in jql.lower():
                jql = jql.lower().split("jql:", 1)[1].strip()

            # If it's still conversational or clearly not JQL, mark as invalid
            if any(jql.lower().startswith(phrase) for phrase in ["i'm sorry", "i cannot", "as an ai"]) or \
               not any(op in jql for op in ["=", "~", "!=", ">", "<", ">=", "<=", "in", "not in", "is", "is not", "and", "or"]) and not ("order by" in jql.lower()):
                return "INVALID_JQL"

            return jql
        except APIError as e:
            logger.error(f"OpenAI API Error in convert_nl_to_jql: {e}")
            return "INVALID_JQL" # Return specific string on API error
        except Exception as e:
            logger.error(f"Unexpected error in convert_nl_to_jql: {e}")
            return "INVALID_JQL" # Return specific string on unexpected error

    def analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of a comment."""
        prompt = f"""Analyze the sentiment of this Jira comment and provide a brief assessment:
        Comment: {text}
        
        Please provide:
        1. Overall sentiment (Positive/Negative/Neutral)
        2. Key emotional indicators
        3. Suggested response approach
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "Unable to analyze sentiment at this time."

    def predict_ticket_priority(self, summary: str, description: str, comments: list) -> dict:
        """Predict ticket priority and provide reasoning."""
        comments_text = "\n".join([f"Comment by {c['author']['displayName']}: {c['body']}" for c in comments])
        prompt = f"""Analyze this Jira ticket and predict its priority level:
        Summary: {summary}
        Description: {description}
        Comments: {comments_text}
        
        Please provide:
        1. Priority level (Highest/High/Medium/Low/Lowest)
        2. Confidence score (0-100%)
        3. Key factors influencing the priority
        4. Recommended actions
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return {
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error predicting ticket priority: {e}")
            return {"error": "Unable to predict priority at this time"}

    def detect_duplicate_tickets(self, current_ticket: dict, existing_tickets: list) -> list:
        """Detect potential duplicate tickets."""
        current_content = f"{current_ticket['summary']} {current_ticket['description']}"
        existing_content = [f"{t['summary']} {t['description']}" for t in existing_tickets]
        
        prompt = f"""Analyze if this ticket might be a duplicate of any existing tickets:
        Current Ticket: {current_content}
        
        Existing Tickets:
        {json.dumps(existing_content, indent=2)}
        
        Please provide:
        1. List of potential duplicates with similarity scores
        2. Key matching points
        3. Recommended action
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return {
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error detecting duplicate tickets: {e}")
            return {"error": "Unable to detect duplicates at this time"}

    def suggest_assignee(self, ticket: dict, team_members: list) -> dict:
        """Suggest the best assignee based on ticket content and team workload."""
        team_info = "\n".join([f"{m['name']}: {m['expertise']}, Current workload: {m['workload']}" for m in team_members])
        prompt = f"""Suggest the best assignee for this ticket based on expertise and workload:
        Ticket: {json.dumps(ticket, indent=2)}
        
        Team Members:
        {team_info}
        
        Please provide:
        1. Recommended assignee
        2. Reasoning
        3. Alternative suggestions
        4. Workload impact assessment
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return {
                "suggestion": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error suggesting assignee: {e}")
            return {"error": "Unable to suggest assignee at this time"}

    def predict_resolution_time(self, ticket: dict, historical_data: list) -> dict:
        """Predict ticket resolution time based on historical data."""
        historical_info = "\n".join([f"Ticket {t['key']}: {t['type']}, Resolution time: {t['resolution_time']}" for t in historical_data])
        prompt = f"""Predict the resolution time for this ticket based on historical data:
        Current Ticket: {json.dumps(ticket, indent=2)}
        
        Historical Data:
        {historical_info}
        
        Please provide:
        1. Predicted resolution time
        2. Confidence level
        3. Key factors affecting the prediction
        4. Risk factors
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            return {
                "prediction": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error predicting resolution time: {e}")
            return {"error": "Unable to predict resolution time at this time"}

    def generate_auto_comment(self, ticket: dict, action: str) -> str:
        """Generate an automatic comment based on ticket action."""
        prompt = f"""Generate an appropriate comment for this ticket action:
        Ticket: {json.dumps(ticket, indent=2)}
        Action: {action}
        
        Please provide a professional and informative comment that:
        1. Explains the action taken
        2. Provides relevant context
        3. Includes next steps if applicable
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating auto comment: {e}")
            return "Unable to generate automatic comment at this time."

    def analyze_workload_distribution(self, team_members: list, tickets: list) -> dict:
        """Analyze team workload distribution and suggest optimizations."""
        team_info = "\n".join([f"{m['name']}: {m['workload']} tickets" for m in team_members])
        ticket_info = "\n".join([f"{t['key']}: {t['priority']}, {t['status']}" for t in tickets])
        
        prompt = f"""Analyze the current workload distribution and suggest optimizations:
        Team Members:
        {team_info}
        
        Current Tickets:
        {ticket_info}
        
        Please provide:
        1. Current workload analysis
        2. Identified bottlenecks
        3. Suggested redistributions
        4. Risk assessment
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return {
                "analysis": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing workload distribution: {e}")
            return {"error": "Unable to analyze workload distribution at this time"}

    def extract_action_entities(self, message: str) -> dict:
        """
        Uses OpenAI to extract the user's intent (action), issue key, and comment text from a message.
        Returns a dict: {action, issue_key, comment_text}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a Jira bot that helps users manage their tickets. 
                    Extract the following from the user's message and return ONLY a valid JSON object:
                    - action: One of these exact values: 'add_comment', 'fetch_comment', 'change_status', 'search', 'analyze', or 'unknown'
                    - issue_key: The Jira ticket number (e.g., CBT-6030, ABC-123)
                    - comment_text: The text of the comment if present
                    
                    If the message is asking for ticket analysis, summary, or details, use 'analyze' as the action.
                    Example messages that should trigger 'analyze':
                    - "contents and a short summary on CBT-5829"
                    - "tell me about CBT-5829"
                    - "what is CBT-5829"
                    - "analyze CBT-5829"
                    - "give me details of CBT-5829"
                    
                    Example response format:
                    {"action": "analyze", "issue_key": "CBT-6030", "comment_text": ""}
                    """},
                    {"role": "user", "content": message}
                ]
            )
            
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                entities = json.loads(content)
                # Validate required fields
                if not isinstance(entities, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure all required fields exist
                required_fields = ['action', 'issue_key', 'comment_text']
                for field in required_fields:
                    if field not in entities:
                        entities[field] = ''
                
                return entities
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in extract_action_entities: {e}")
                logger.error(f"Raw response: {content}")
                return {'action': 'unknown', 'issue_key': '', 'comment_text': ''}
                
        except Exception as e:
            logger.error(f"Error in extract_action_entities: {e}")
            return {'action': 'unknown', 'issue_key': '', 'comment_text': ''}

    def complete(self, prompt, max_tokens=1500, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            return "" 