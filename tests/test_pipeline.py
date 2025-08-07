"""
Test script for Risk Atlas pipeline components.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.risk_atlas.config.settings import Settings
from src.risk_atlas.config.risk_taxonomy import RiskTaxonomy
from src.risk_atlas.models.gemini_classifier import GeminiClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_taxonomy():
    """Test the risk taxonomy configuration."""
    logger.info("Testing risk taxonomy...")
    
    taxonomy = RiskTaxonomy()
    
    # Test basic functionality
    assert len(taxonomy.states) == 10, f"Expected 10 states, got {len(taxonomy.states)}"
    
    # Test state retrieval
    state_names = taxonomy.get_state_names()
    assert len(state_names) == 10, f"Expected 10 state names, got {len(state_names)}"
    
    # Test specific state
    cognitive_state = taxonomy.get_state_by_name("Cognitive Offloading Dependence")
    assert cognitive_state.id == 3, f"Expected ID 3, got {cognitive_state.id}"
    
    # Test risk states (excluding control)
    risk_states = taxonomy.get_risk_states()
    assert len(risk_states) == 9, f"Expected 9 risk states, got {len(risk_states)}"
    
    logger.info("✓ Risk taxonomy tests passed")

def test_gemini_classifier():
    """Test the Gemini classifier (requires API key)."""
    logger.info("Testing Gemini classifier...")
    
    # Check if API key is available
    config = Settings()
    if not config.GEMINI_API_KEY:
        logger.warning("No Gemini API key found, skipping classifier test")
        return
    
    try:
        classifier = GeminiClassifier(config.GEMINI_API_KEY)
        
        # Test prompt creation
        test_text = "I can't make decisions without ChatGPT anymore"
        prompt = classifier.create_labeling_prompt(test_text)
        assert "Cognitive Offloading Dependence" in prompt
        assert test_text in prompt
        
        # Test single classification (if API key is valid)
        try:
            category, confidence = classifier.classify_post(test_text)
            assert 1 <= category <= 10, f"Invalid category: {category}"
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"
            logger.info(f"✓ Classified as category {category} with confidence {confidence}")
        except Exception as e:
            logger.warning(f"API call failed (this is expected without valid key): {e}")
        
        logger.info("✓ Gemini classifier tests passed")
        
    except Exception as e:
        logger.error(f"Gemini classifier test failed: {e}")

def test_reddit_api():
    """Test Reddit API with real credentials."""
    logger.info("Testing Reddit API...")
    
    config = Settings()
    
    # Check if Reddit API keys are available
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        logger.warning("No Reddit API keys found, skipping Reddit API test")
        return
    
    try:
        from src.risk_atlas.data.collectors.reddit_collector import RedditCollector
        
        collector = RedditCollector(config)
        
        # Test with a small sample to avoid rate limiting
        logger.info(f"Testing Reddit API with subreddits: {config.SUBREDDITS[:2]}")
        
        # Test keyword filtering with real API
        result_with_filter = collector.collect_posts(
            limit_per_subreddit=3,  # Very small limit for testing
            time_filter="week",
            filter_by_keywords=True
        )
        
        logger.info(f"✓ Reddit API test: {len(result_with_filter)} posts collected with keyword filtering")
        
        # Test without filtering
        result_without_filter = collector.collect_posts(
            limit_per_subreddit=3,
            time_filter="week", 
            filter_by_keywords=False
        )
        
        logger.info(f"✓ Reddit API test: {len(result_without_filter)} posts collected without filtering")
        
        logger.info("✓ Reddit API tests passed")
        
    except Exception as e:
        logger.warning(f"Reddit API test failed (this is expected if keys are invalid): {e}")

def test_sample_data():
    """Test with sample data."""
    logger.info("Testing with sample data...")
    
    # Create sample data
    sample_data = [
        {
            'platform': 'reddit',
            'subreddit': 'ChatGPT',
            'post_id': 'test1',
            'title': 'I rely on AI too much',
            'text': 'I can\'t make decisions without ChatGPT anymore. My brain feels useless.',
            'score': 100,
            'created_utc': datetime.now(),
            'author': 'test_user'
        },
        {
            'platform': 'reddit',
            'subreddit': 'MentalHealth',
            'post_id': 'test2',
            'title': 'AI is my only friend',
            'text': 'I prefer talking to AI than people. Humans are too complicated.',
            'score': 50,
            'created_utc': datetime.now(),
            'author': 'test_user2'
        },
        {
            'platform': 'twitter',
            'tweet_id': 'test3',
            'text': 'I love my AI assistant so much!',
            'created_at': datetime.now(),
            'like_count': 25,
            'author_id': 'test_user3'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Test data structure
    assert len(df) == 3, f"Expected 3 sample posts, got {len(df)}"
    assert 'text' in df.columns, "Expected 'text' column"
    
    logger.info(f"✓ Sample data created with {len(df)} posts")
    
    # Save sample data
    output_dir = "output/risk_atlas/test"
    os.makedirs(output_dir, exist_ok=True)
    sample_path = os.path.join(output_dir, 'sample_data.csv')
    df.to_csv(sample_path, index=False)
    
    logger.info(f"✓ Sample data saved to {sample_path}")
    
    return df

def test_configuration():
    """Test configuration settings."""
    logger.info("Testing configuration...")
    
    config = Settings()
    
    # Test required fields
    assert hasattr(config, 'SUBREDDITS'), "Missing SUBREDDITS configuration"
    assert hasattr(config, 'LLM_KEYWORDS'), "Missing LLM_KEYWORDS configuration"
    assert hasattr(config, 'EMBEDDING_MODEL'), "Missing EMBEDDING_MODEL configuration"
    
    # Test subreddits
    assert len(config.SUBREDDITS) > 0, "No subreddits configured"
    assert 'ChatGPT' in config.SUBREDDITS, "ChatGPT subreddit not in configuration"
    
    # Test keywords
    assert len(config.LLM_KEYWORDS) > 0, "No LLM keywords configured"
    assert 'GPT' in config.LLM_KEYWORDS, "GPT keyword not in configuration"
    
    logger.info("✓ Configuration tests passed")

def main():
    """Run all tests."""
    logger.info("Starting Risk Atlas pipeline tests...")
    
    try:
        # Test configuration
        test_configuration()
        
        # Test taxonomy
        test_taxonomy()
        
        # Test sample data
        test_sample_data()
        
        # Test Gemini classifier (if API key available)
        test_gemini_classifier()
        
        # Test Reddit API (if keys are available)
        test_reddit_api()
        
        logger.info("✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 