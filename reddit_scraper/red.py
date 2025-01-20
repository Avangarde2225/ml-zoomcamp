import praw
import pandas as pd
import re
import time
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reddit_scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()  # This will output to console
    ]
)

# Constants
CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')

if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
    raise ValueError("Missing Reddit API credentials in .env file")

POSTS_LIMIT = 1000
COMMENT_BATCH_SIZE = 100  # Number of comments to process before sleeping
RATE_LIMIT_SLEEP = 2  # Seconds to sleep between batches

SUBREDDITS = [
    "dropship",
    "Entrepreneur",
    "smallbusiness",
    "startups",
    "Business_Ideas",
    "SideProject",
    "SideHustle",
    "Flipping",
]

IDEA_PATTERNS = [
    r"(?:business idea[s]?:?\s*)(.*?)(?=\n|$|pros?:|cons?:)",
    r"(?:startup idea[s]?:?\s*)(.*?)(?=\n|$|pros?:|cons?:)",
    r"(?:my idea is:?\s*)(.*?)(?=\n|$|pros?:|cons?:)",
    r"(?:thinking about:?\s*)(.*?)(?=\n|$|pros?:|cons?:)",
    r"(?:planning to:?\s*)(.*?)(?=\n|$|pros?:|cons?:)",
    r"(?:want to start:?\s*)(.*?)(?=\n|$|pros?:|cons?:)"
]

SENTIMENT_INDICATORS = {
    'positive': [
        'advantage', 'benefit', 'good', 'great', 'positive', 'profitable',
        'success', 'opportunity', 'efficient', 'effective', 'easy',
        'profitable', 'scalable', 'potential', 'growth', 'recommend',
        'worth', 'helpful', 'useful', 'valuable', 'affordable'
    ],
    'negative': [
        'disadvantage', 'drawback', 'bad', 'negative', 'difficult', 'hard',
        'expensive', 'risky', 'challenge', 'problem', 'issue', 'concern',
        'careful', 'avoid', 'costly', 'complicated', 'time-consuming',
        'competitive', 'saturated', 'warning', 'careful'
    ]
}

def create_reddit_instance() -> praw.Reddit:
    """Create and return authenticated Reddit instance."""
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        logging.info("Successfully created Reddit instance")
        return reddit
    except Exception as e:
        logging.error(f"Failed to create Reddit instance: {str(e)}")
        raise

def extract_pros_cons(text: str) -> Tuple[str, str]:
    """Extract pros and cons from text."""
    pros_pattern = r"(?:\bpros?\b:\s*)(.*?)(?=\bcons?\b:|$)"
    cons_pattern = r"(?:\bcons?\b:\s*)(.*?)(?=$)"
    
    try:
        pros_matches = re.search(pros_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        cons_matches = re.search(cons_pattern, text, flags=re.IGNORECASE | re.DOTALL)
        
        pros = pros_matches.group(1).strip() if pros_matches else ""
        cons = cons_matches.group(1).strip() if cons_matches else ""
        
        return pros, cons
    except Exception as e:
        logging.warning(f"Error extracting pros/cons: {str(e)}")
        return "", ""

def extract_business_ideas(text: str) -> str:
    """Extract business ideas from text using multiple patterns."""
    try:
        for pattern in IDEA_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""
    except Exception as e:
        logging.warning(f"Error extracting business idea: {str(e)}")
        return ""

def get_comments(post: praw.models.Submission) -> str:
    """Get all comments from a post with rate limiting."""
    try:
        post.comments.replace_more(limit=None)
        comments_text = []
        for i, comment in enumerate(post.comments.list()):
            comments_text.append(comment.body)
            if i % COMMENT_BATCH_SIZE == 0 and i > 0:
                time.sleep(RATE_LIMIT_SLEEP)
        return " ".join(comments_text)
    except Exception as e:
        logging.warning(f"Error getting comments for post {post.id}: {str(e)}")
        return ""

def extract_idea_context(text: str, idea: str, window_size: int = 200) -> Dict[str, str]:
    """
    Extract the surrounding context of an identified business idea.
    Returns both the preceding and following text around the idea.
    """
    try:
        if not idea:
            return {"context": text}
            
        # Find the position of the idea in the text
        idea_pos = text.lower().find(idea.lower())
        if idea_pos == -1:
            return {"context": text}
            
        # Extract text before and after the idea
        start = max(0, idea_pos - window_size)
        end = min(len(text), idea_pos + len(idea) + window_size)
        
        context = text[start:end]
        
        return {
            "context": context,
            "idea": idea,
            "sentiment_indicators": analyze_sentiment(context)
        }
    except Exception as e:
        logging.warning(f"Error extracting context: {str(e)}")
        return {"context": text}

def analyze_sentiment(text: str) -> Dict[str, List[str]]:
    """
    Analyze text for positive and negative sentiment indicators.
    Returns found indicators for human review.
    """
    text_lower = text.lower()
    found_indicators = {
        'positive': [],
        'negative': []
    }
    
    # Find sentiment indicators
    for sentiment, words in SENTIMENT_INDICATORS.items():
        for word in words:
            if word in text_lower:
                found_indicators[sentiment].append(word)
    
    return found_indicators

def process_subreddit(reddit: praw.Reddit, subreddit_name: str) -> List[Dict]:
    """Process a single subreddit and return its posts data."""
    subreddit = reddit.subreddit(subreddit_name)
    subreddit_posts = []
    
    logging.info(f"Starting to process subreddit: {subreddit_name}")
    try:
        for i, post in enumerate(subreddit.new(limit=POSTS_LIMIT)):
            try:
                logging.info(f"Processing post {i+1}/{POSTS_LIMIT} from {subreddit_name}: {post.id}")
                
                combined_text = f"{post.title}\n\n{post.selftext}"
                comments_text = get_comments(post)
                full_text = f"{combined_text}\n\n{comments_text}"
                
                idea = extract_business_ideas(full_text)
                if idea:
                    logging.info(f"Found business idea in post {post.id}: {idea[:100]}...")
                
                idea_analysis = extract_idea_context(full_text, idea)
                pros, cons = extract_pros_cons(full_text)

                post_info = {
                    "subreddit": subreddit_name,
                    "business_idea": idea,
                    "idea_context": idea_analysis["context"],
                    "positive_indicators": ", ".join(idea_analysis["sentiment_indicators"]["positive"]),
                    "negative_indicators": ", ".join(idea_analysis["sentiment_indicators"]["negative"]),
                    "explicit_pros": pros,
                    "explicit_cons": cons,
                    "url": post.url,
                    "title": post.title,
                    "selftext": post.selftext,
                    "comments": comments_text,
                    "created_utc": datetime.fromtimestamp(post.created_utc, tz=pytz.UTC).isoformat()
                }
                subreddit_posts.append(post_info)
                
                # Rate limiting
                time.sleep(RATE_LIMIT_SLEEP)
                
            except Exception as e:
                logging.error(f"Error processing post {post.id}: {str(e)}")
                continue
                
        logging.info(f"Completed processing subreddit {subreddit_name}. Found {len(subreddit_posts)} posts.")
                
    except Exception as e:
        logging.error(f"Error processing subreddit {subreddit_name}: {str(e)}")
    
    return subreddit_posts

def main():
    """Main function to run the Reddit scraper."""
    try:
        reddit = create_reddit_instance()
        all_posts_data = []
        
        for subreddit_name in SUBREDDITS:
            subreddit_posts = process_subreddit(reddit, subreddit_name)
            all_posts_data.extend(subreddit_posts)
            
            # Save intermediate results
            df = pd.DataFrame(all_posts_data)
            df.to_csv(f"reddit_business_ideas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            
        logging.info("Scraping completed successfully!")
        
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()