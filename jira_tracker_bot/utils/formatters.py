"""
Formatting utilities for the Jira Tracker Bot.
"""

from datetime import datetime
from typing import Dict, Any

def format_ticket_info(ticket: Dict[str, Any]) -> str:
    """Format ticket information for display."""
    return f"""
Ticket: {ticket['key']}
Title: {ticket['fields']['summary']}
Status: {ticket['fields']['status']['name']}
Assignee: {ticket['fields'].get('assignee', {}).get('displayName', 'Unassigned')}
Priority: {ticket['fields'].get('priority', {}).get('name', 'Not set')}
Created: {datetime.fromisoformat(ticket['fields']['created'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}
Updated: {datetime.fromisoformat(ticket['fields']['updated'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}
"""

def format_ai_analysis(analysis: Dict[str, Any]) -> str:
    """Format AI analysis results for display."""
    return f"""
AI Analysis:
------------
Summary: {analysis.get('summary', 'No summary available')}
Sentiment: {analysis.get('sentiment', 'Neutral')}
Risk Level: {analysis.get('risk_level', 'Unknown')}
Predicted Resolution: {analysis.get('predicted_resolution', 'Not available')}
Key Points:
{chr(10).join(f'- {point}' for point in analysis.get('key_points', []))}
""" 