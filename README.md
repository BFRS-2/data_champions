# Jira Tracker Bot

An AI-powered Jira ticket tracking and analysis tool that helps teams manage and analyze their Jira tickets efficiently.

## Features

- Automated Jira ticket tracking
- AI-powered ticket analysis
- Telegram bot integration
- Customizable reporting
- Ticket prioritization
- Team workload analysis

## Project Structure

```
jira_tracker_bot/
├── core/               # Core functionality
│   ├── bot.py         # Main bot implementation
│   ├── jira_client.py # Jira API integration
│   └── openai_client.py # OpenAI integration
├── utils/             # Utility functions
│   ├── formatters.py  # Data formatting utilities
│   └── validators.py  # Input validation
├── config/            # Configuration files
├── data/             # Data storage
└── cli.py            # Command-line interface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jira-tracker-bot.git
cd jira-tracker-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
JIRA_API_TOKEN=your_jira_api_token
JIRA_EMAIL=your_jira_email
JIRA_URL=your_jira_instance_url
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

## Usage

1. Run the bot:
```bash
python -m jira_tracker_bot.cli
```

2. Use the Telegram bot:
- Start a chat with your bot
- Use commands like `/help` to see available options
- Track tickets and get AI-powered insights

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

---

## Running the Bot on Another System

To run the Jira Tracker Bot on a different machine (e.g., a new server, your colleague's computer, or a cloud VM), follow these steps:

1. **Clone or Copy the Repository**
   - If you have access to the repository, clone it:
     ```bash
     git clone https://github.com/yourusername/jira-tracker-bot.git
     cd jira-tracker-bot
     ```
   - Or, copy the project folder to the new system.

2. **Set Up Python and Virtual Environment**
   - Ensure Python 3.8+ is installed on the new system.
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

3. **Install Dependencies**
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Or, to install as a package (if needed):
     ```bash
     pip install .
     ```

4. **Configure Environment Variables**
   - Copy your existing `.env` file, or create a new one in the project root with the following:
     ```
     JIRA_API_TOKEN=your_jira_api_token
     JIRA_EMAIL=your_jira_email
     JIRA_URL=your_jira_instance_url
     OPENAI_API_KEY=your_openai_api_key
     TELEGRAM_BOT_TOKEN=your_telegram_bot_token
     ```
   - Make sure all values are correct for the new environment.

5. **Run the Bot**
   - Start the bot using:
     ```bash
     python -m jira_tracker_bot.cli
     ```
   - Or, use any provided CLI or scripts as described above.

6. **Troubleshooting**
   - If you encounter issues:
     - Ensure all environment variables are set and correct.
     - Check that all dependencies are installed.
     - Review logs in the `logs/` directory for error messages.
     - Make sure the system has internet access for Jira, OpenAI, and Telegram APIs.

7. **Telegram Bot Setup**
   - If running on a new Telegram account or bot, update the `TELEGRAM_BOT_TOKEN` in your `.env` file.
   - Start a chat with your bot and use `/help` to verify it is running.

--- 