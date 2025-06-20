import os
import logging
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class JiraClient:
    def __init__(self, jira_domain: str, jira_email: str, jira_api_token: str):
        self.jira_domain = jira_domain
        self.jira_email = jira_email
        self.jira_api_token = jira_api_token
        self.auth = (self.jira_email, self.jira_api_token)
        self.headers = {"Accept": "application/json"}

    def get_jira_issue(self, issue_key: str):
        """Fetch issue details from Jira."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error fetching issue {issue_key}: {e}")
            return None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error fetching issue {issue_key}: {e}")
            return None, str(e)

    def get_comments(self, issue_key: str):
        """Fetch comments for a Jira issue."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}/comment"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error fetching comments for {issue_key}: {e}")
            return None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error fetching comments for {issue_key}: {e}")
            return None, str(e)

    def get_my_jira_profile(self):
        """Fetches the current user's Jira profile to get display name and email."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/myself"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error fetching profile: {e}")
            return None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error fetching profile: {e}")
            return None, str(e)

    def find_similar_tickets(self, issue_data: dict):
        """Find similar tickets using Jira API and text similarity."""
        try:
            jql = f"project = {issue_data['fields']['project']['key']}"
            url = f"https://{self.jira_domain}/rest/api/3/search"
            params = {"jql": jql, "maxResults": 50}
            response = requests.get(url, headers=self.headers, auth=self.auth, params=params)
            response.raise_for_status()
            recent_tickets = response.json().get('issues', [])

            # Exclude the current issue from similar tickets
            filtered_tickets = [ticket for ticket in recent_tickets if ticket['key'] != issue_data['key']]
            
            return filtered_tickets
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error searching for similar tickets: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching for similar tickets: {e}")
            return []

    def search_issues_by_jql(self, jql_query: str):
        """Searches Jira issues using a JQL query."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/search"
            params = {"jql": jql_query, "maxResults": 50}
            response = requests.get(url, headers=self.headers, auth=self.auth, params=params)
            response.raise_for_status()
            return response.json().get('issues', []), None
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error during JQL search: {e}")
            return [], str(e)
        except Exception as e:
            logger.error(f"Unexpected error during JQL search: {e}")
            return [], str(e)

    def change_issue_status(self, issue_key: str, transition_name: str):
        """Changes the status of a Jira issue."""
        try:
            # First, get available transitions for the issue
            transitions_url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}/transitions"
            transitions_response = requests.get(transitions_url, headers=self.headers, auth=self.auth)
            transitions_response.raise_for_status()
            transitions_data = transitions_response.json()
            
            transition_id = None
            for t in transitions_data.get('transitions', []):
                if t['name'].lower() == transition_name.lower():
                    transition_id = t['id']
                    break

            if not transition_id:
                return False, f"Status '{transition_name}' not found or not a valid transition for {issue_key}."

            # Then, perform the transition
            transition_url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}/transitions"
            payload = {"transition": {"id": transition_id}}
            action_response = requests.post(transition_url, headers=self.headers, auth=self.auth, json=payload)
            action_response.raise_for_status()
            return True, f"Status of {issue_key} changed to '{transition_name}'."
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error changing status for {issue_key}: {e}")
            return False, f"Failed to change status: {e}"
        except Exception as e:
            logger.error(f"Unexpected error changing status for {issue_key}: {e}")
            return False, f"An unexpected error occurred: {e}"

    def add_comment_to_issue(self, issue_key: str, comment_body: str):
        """Adds a comment to a Jira issue."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}/comment"
            payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": comment_body
                                }
                            ]
                        }
                    ]
                }
            }
            response = requests.post(url, headers=self.headers, auth=self.auth, json=payload)
            response.raise_for_status()
            return True, f"Comment added to {issue_key}."
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error adding comment to {issue_key}: {e}")
            return False, f"Failed to add comment: {e}"
        except Exception as e:
            logger.error(f"Unexpected error adding comment to {issue_key}: {e}")
            return False, f"An unexpected error occurred: {e}"

    def assign_issue(self, issue_key: str, assignee_id: str):
        """Assigns a Jira issue to a user by account ID."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/issue/{issue_key}/assignee"
            payload = {"accountId": assignee_id}
            response = requests.put(url, headers=self.headers, auth=self.auth, json=payload)
            response.raise_for_status()
            return True, f"Issue {issue_key} assigned to {assignee_id}."
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error assigning {issue_key} to {assignee_id}: {e}")
            return False, f"Failed to assign issue: {e}"
        except Exception as e:
            logger.error(f"Unexpected error assigning {issue_key} to {assignee_id}: {e}")
            return False, f"An unexpected error occurred: {e}"

    def get_user_account_id(self, email_address: str):
        """Searches for a user by email address and returns their account ID."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/user/search"
            params = {"query": email_address}
            response = requests.get(url, headers=self.headers, auth=self.auth, params=params)
            response.raise_for_status()
            users = response.json()
            if users and users[0]['emailAddress'].lower() == email_address.lower():
                return users[0]['accountId'], None
            return None, f"User with email {email_address} not found."
        except requests.exceptions.RequestException as e:
            logger.error(f"Jira API Error searching for user {email_address}: {e}")
            return None, f"Failed to search for user: {e}"
        except Exception as e:
            logger.error(f"Unexpected error searching for user {email_address}: {e}")
            return None, f"An unexpected error occurred: {e}"

    def create_issue(self, project_key, summary, description, issue_type, parent=None):
        """Create a Jira issue (Epic, Story, Sub-task) with optional parent for hierarchy."""
        try:
            url = f"https://{self.jira_domain}/rest/api/3/issue"
            fields = {
                "project": {"key": project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description or ""
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {"name": issue_type}
            }
            # For Epics, set Epic Name (required by Jira Cloud)
            if issue_type.lower() == "epic":
                fields["customfield_10009"] = summary  # Epic Name (CBT project)
            # For Sub-tasks, set parent
            if issue_type.lower() == "subtask" and parent:
                fields["parent"] = {"key": parent}
            # For Stories with Epic parent, set Epic Link (if parent is an Epic)
            if parent and issue_type.lower() == "story":
                # Set Epic Link to parent Epic
                fields["customfield_10008"] = parent
            payload = {"fields": fields}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, auth=self.auth, json=payload)
            response.raise_for_status()
            issue_key = response.json().get("key")
            return issue_key, None
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Jira API Error creating issue: {e} | Response: {e.response.text}")
                return None, f"{str(e)} | {e.response.text}"
            logger.error(f"Jira API Error creating issue: {e}")
            return None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error creating issue: {e}")
            return None, str(e) 