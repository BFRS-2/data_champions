from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jira-tracker-bot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered Jira ticket tracking and analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jira-tracker-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jira>=3.5.1",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "rich>=10.0.0",
        "typer>=0.4.0",
        "requests>=2.31.0",
        "nltk>=3.8.1",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "python-telegram-bot>=20.0",
        "apscheduler>=3.10.0"
    ],
    entry_points={
        "console_scripts": [
            "jira=jira_tracker_bot.cli:main",
        ],
    },
) 