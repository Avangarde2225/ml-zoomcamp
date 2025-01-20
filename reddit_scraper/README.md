# Reddit Business Idea Evaluator 💡

## Overview
This tool scrapes business-related subreddits to collect, analyze, and evaluate business ideas. It extracts business ideas, their pros and cons, and performs sentiment analysis to help entrepreneurs make data-driven decisions.

## Features
- 🤖 Multi-subreddit scraping
- 📊 Sentiment analysis visualization
- 🔍 Business idea extraction
- 💬 Comment analysis
- 📈 Trend identification
- 🗂 Structured data storage

## Prerequisites
- Python 3.9+
- Reddit API credentials
- Docker (optional)

## Installation

1. Clone the repository:

```
bash
git clone <repository-url>
cd reddit_scraper
```

2. Create virtual environment:

```
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Add your Reddit API credentials:
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

## Usage

1. Run the Reddit scraper:
```bash
python red.py
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## Project Structure
```
reddit_scraper/
├── app.py              # Streamlit application
├── red.py              # Reddit scraper
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .env              # Environment variables
├── .env.example      # Example environment variables
├── README.md         # Documentation
├── .gitignore        # Git ignore file
└── task-definition.json  # AWS ECS task definition
```

## Data Collection
The scraper collects data from the following subreddits:
- r/dropship
- r/Entrepreneur
- r/smallbusiness
- r/startups
- r/Business_Ideas
- r/SideProject
- r/SideHustle
- r/Flipping

## Features Details

### Business Idea Extraction
- Identifies business ideas from posts and comments
- Extracts contextual information
- Captures related discussions

### Sentiment Analysis
- Identifies positive and negative indicators
- Analyzes sentiment around business ideas
- Provides trend visualization

### Data Visualization
- Interactive data tables
- Sentiment distribution charts
- Time-based filtering
- Subreddit-based filtering

## License
MIT License
