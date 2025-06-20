# Jira Tracker Bot

An AI-powered Jira ticket tracking and analysis tool that provides smart insights, sentiment analysis, and proactive notifications for your Jira tickets.

## Features

- 🔍 Smart ticket analysis and summarization
- 😊 Sentiment analysis of ticket comments
- ⚠️ Proactive notifications for mentions and stale tickets
- 📊 AI-powered predictions for resolution times
- ⚡ Real-time ticket monitoring
- 🎯 Customizable notification preferences

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jira-tracker-bot.git
cd jira-tracker-bot
```

2. Install the package:
```bash
pip install -e .
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```env
JIRA_URL=your_jira_url
JIRA_EMAIL=your_email
JIRA_API_TOKEN=your_api_token
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
jira_tracker_bot/
├── jira_tracker_bot/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── jira_client.py
│   │   └── openai_client.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── formatters.py
│   │   └── validators.py
│   ├── config/
│   │   └── __init__.py
│   ├── cli.py
│   └── __init__.py
├── setup.py
└── README.md
```

## Usage

Run the bot using:
```bash
jira-tracker
```

### Commands

- `ticket <ticket_id>`: Get detailed information about a specific ticket
- `search <query>`: Search for tickets matching the query
- `settings`: Configure notification preferences
- `help`: Show available commands and their usage

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 