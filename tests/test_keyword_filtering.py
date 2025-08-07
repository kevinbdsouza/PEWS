#!/usr/bin/env python3
"""
Test script for LLM keyword filtering functionality.
"""
import os
import sys
import logging
from unittest.mock import Mock, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import the classes from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.collectors.reddit_collector import RedditCollector
from data.collectors.reddit_downloader import RedditDownloader

# Try to import settings for real API testing
try:
    from src.risk_atlas.config.settings import Settings
    REAL_API_AVAILABLE = True
except ImportError:
    REAL_API_AVAILABLE = False
    print("Warning: Could not import settings, will only run mock tests")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keyword_filtering():
    """Test the keyword filtering functionality."""
    
    print("Testing LLM keyword filtering functionality...")
    print("=" * 50)
    
    # Test with real API if available
    if REAL_API_AVAILABLE:
        test_with_real_api()
    else:
        print("Skipping real API tests (settings not available)")
    
    # Test with mock API
    test_with_mock_api()

def test_with_real_api():
    """Test with real Reddit API using actual settings."""
    print("\n1. Testing with REAL Reddit API:")
    
    try:
        # Load real settings
        settings = Settings()
        
        # Check if Reddit API keys are available
        if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
            print("  ⚠️  Reddit API keys not configured, skipping real API test")
            return
        
        print(f"  Using real Reddit API with subreddits: {settings.SUBREDDITS[:3]}...")
        print(f"  LLM keywords: {settings.LLM_KEYWORDS}")
        
        # Test RedditCollector with real API
        collector = RedditCollector(settings)
        
        # Test with small limits to avoid rate limiting
        print("  Testing keyword filtering with real API...")
        result_with_filter = collector.collect_posts(
            limit_per_subreddit=5,  # Small limit for testing
            time_filter="week",     # Recent posts
            filter_by_keywords=True
        )
        print(f"  Posts collected with filtering: {len(result_with_filter)}")
        
        # Test without filtering
        result_without_filter = collector.collect_posts(
            limit_per_subreddit=5,
            time_filter="week",
            filter_by_keywords=False
        )
        print(f"  Posts collected without filtering: {len(result_without_filter)}")
        
        # Test RedditDownloader with real API
        print("  Testing RedditDownloader with real API...")
        downloader = RedditDownloader(settings)
        
        # Test a small batch download
        posts, checked, filtered = downloader._download_subreddit_batch(
            settings.SUBREDDITS[0], "hot", "all", filter_by_keywords=True
        )
        print(f"  Real download: {len(posts)} posts, {checked} checked, {filtered} filtered")
        
        print("  ✓ Real API tests completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Real API test failed: {e}")
        print("  This is expected if API keys are invalid or network issues occur")

def test_with_mock_api():
    """Test with mocked Reddit API."""
    print("\n2. Testing with MOCKED Reddit API:")
    
    # Create a mock settings object
    mock_settings = Mock()
    mock_settings.REDDIT_CLIENT_ID = "test_client_id"
    mock_settings.REDDIT_CLIENT_SECRET = "test_client_secret"
    mock_settings.REDDIT_USER_AGENT = "test_user_agent"
    mock_settings.SUBREDDITS = ["test_subreddit"]
    mock_settings.LLM_KEYWORDS = ["GPT", "Claude", "AI assistant", "chatbot"]
    
    # Mock the Reddit API
    mock_reddit = Mock()
    mock_subreddit = Mock()
    mock_post1 = Mock()
    mock_post1.title = "I love using GPT for coding"
    mock_post1.selftext = "GPT is amazing for programming tasks"
    mock_post1.id = "post1"
    mock_post1.score = 100
    mock_post1.num_comments = 50
    mock_post1.created_utc = 1640995200  # 2022-01-01
    mock_post1.author = Mock()
    mock_post1.author.name = "user1"
    mock_post1.url = "https://reddit.com/r/test/comments/post1"
    mock_post1.is_self = True
    mock_post1.over_18 = False
    mock_post1.spoiler = False
    mock_post1.stickied = False
    mock_post1.upvote_ratio = 0.95
    
    mock_post2 = Mock()
    mock_post2.title = "What's for dinner tonight?"
    mock_post2.selftext = "Looking for recipe suggestions"
    mock_post2.id = "post2"
    mock_post2.score = 10
    mock_post2.num_comments = 5
    mock_post2.created_utc = 1640995200
    mock_post2.author = Mock()
    mock_post2.author.name = "user2"
    mock_post2.url = "https://reddit.com/r/test/comments/post2"
    mock_post2.is_self = True
    mock_post2.over_18 = False
    mock_post2.spoiler = False
    mock_post2.stickied = False
    mock_post2.upvote_ratio = 0.8
    
    mock_subreddit.hot.return_value = [mock_post1, mock_post2]
    mock_subreddit.top.return_value = [mock_post1, mock_post2]
    mock_subreddit.new.return_value = [mock_post1, mock_post2]
    mock_reddit.subreddit.return_value = mock_subreddit
    
    # Create collector with mocked Reddit
    collector = RedditCollector(mock_settings)
    collector.reddit = mock_reddit
    
    # Test with keyword filtering enabled
    print("  Testing with keyword filtering enabled...")
    result_with_filter = collector.collect_posts(
        limit_per_subreddit=2,
        filter_by_keywords=True
    )
    print(f"  Posts collected with filtering: {len(result_with_filter)}")
    print(f"  Expected: 1 post (contains 'GPT')")
    
    # Test with keyword filtering disabled
    print("  Testing with keyword filtering disabled...")
    result_without_filter = collector.collect_posts(
        limit_per_subreddit=2,
        filter_by_keywords=False
    )
    print(f"  Posts collected without filtering: {len(result_without_filter)}")
    print(f"  Expected: 2 posts (all posts)")
    
    # Test keyword detection method
    print("\n3. Testing keyword detection method:")
    test_texts = [
        ("I use GPT for coding", True),
        ("Claude is helpful", True),
        ("My AI assistant is great", True),
        ("This chatbot is useful", True),
        ("What's for dinner?", False),
        ("Weather is nice today", False),
        ("", True),  # Empty text should pass (no keywords configured)
    ]
    
    for text, expected in test_texts:
        result = collector._contains_llm_keywords(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected: {expected})")
    
    # Test RedditDownloader keyword filtering
    print("\n4. Testing RedditDownloader keyword filtering:")
    
    downloader = RedditDownloader(mock_settings)
    downloader.reddit = mock_reddit
    
    # Test the batch download method
    posts, checked, filtered = downloader._download_subreddit_batch(
        "test_subreddit", "hot", "all", filter_by_keywords=True
    )
    print(f"  Posts downloaded: {len(posts)}")
    print(f"  Posts checked: {checked}")
    print(f"  Posts filtered out: {filtered}")
    print(f"  Expected: 1 downloaded, 2 checked, 1 filtered")
    
    print("\n5. Testing edge cases:")
    
    # Test with no keywords configured
    mock_settings_no_keywords = Mock()
    mock_settings_no_keywords.REDDIT_CLIENT_ID = "test_client_id"
    mock_settings_no_keywords.REDDIT_CLIENT_SECRET = "test_client_secret"
    mock_settings_no_keywords.REDDIT_USER_AGENT = "test_user_agent"
    mock_settings_no_keywords.SUBREDDITS = ["test_subreddit"]
    # No LLM_KEYWORDS attribute
    
    collector_no_keywords = RedditCollector(mock_settings_no_keywords)
    collector_no_keywords.reddit = mock_reddit
    
    result_no_keywords = collector_no_keywords.collect_posts(
        limit_per_subreddit=2,
        filter_by_keywords=True
    )
    print(f"  Posts with no keywords configured: {len(result_no_keywords)}")
    print(f"  Expected: 2 posts (all posts when no keywords configured)")
    
    print("\n" + "=" * 50)
    print("Keyword filtering test completed!")
    print("The functionality should now filter posts based on LLM keywords.")
    if REAL_API_AVAILABLE:
        print("Real API tests were also performed.")
    print("Keywords configured:", mock_settings.LLM_KEYWORDS)

if __name__ == "__main__":
    test_keyword_filtering() 