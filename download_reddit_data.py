#!/usr/bin/env python3
"""
Standalone script to download Reddit data for Risk Atlas project.
Respects Reddit API rate limits: 1000 requests per 10 minutes, 100 posts per request.
"""
import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.risk_atlas.config.settings import Settings
from src.risk_atlas.data.collectors.reddit_downloader import RedditDownloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Reddit data download."""
    parser = argparse.ArgumentParser(description='Download Reddit data for Risk Atlas')
    parser.add_argument('--target-posts', type=int, default=1000000,
                       help='Target number of posts to download (default: 1,000,000)')
    parser.add_argument('--output-dir', type=str, default='data/reddit_downloads',
                       help='Output directory for downloaded data (default: data/reddit_downloads)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from existing download')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics of existing download')
    parser.add_argument('--no-keyword-filter', action='store_true',
                       help='Disable LLM keyword filtering (download all posts)')

    args = parser.parse_args()
    #args.no_keyword_filter = True

    # Load configuration
    config = Settings()
    
    # Validate configuration
    if not config.validate_config():
        logger.error("Invalid configuration. Please check your Reddit API keys.")
        logger.info("Get FREE Reddit API keys: https://www.reddit.com/prefs/apps")
        return 1
    
    try:
        # Initialize downloader
        downloader = RedditDownloader(config)
        
        # Check for existing data
        data_file = os.path.join(args.output_dir, "reddit_posts.csv")
        
        if args.stats_only:
            if os.path.exists(data_file):
                stats = downloader.get_download_stats(data_file)
                print("\nReddit Download Statistics:")
                print("=" * 40)
                print(f"Total Posts: {stats.get('total_posts', 0):,}")
                print(f"Unique Subreddits: {stats.get('unique_subreddits', 0)}")
                print(f"Date Range: {stats.get('date_range', {}).get('earliest', 'N/A')} to {stats.get('date_range', {}).get('latest', 'N/A')}")
                print(f"Average Score: {stats.get('avg_score', 0):.1f}")
                print(f"Average Comments: {stats.get('avg_comments', 0):.1f}")
                
                print("\nSubreddit Distribution (top 10):")
                subreddit_dist = stats.get('subreddit_distribution', {})
                for subreddit, count in sorted(subreddit_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  r/{subreddit}: {count:,} posts")
            else:
                logger.error(f"No existing data found at {data_file}")
                return 1
        else:
            # Start download
            filter_by_keywords = not args.no_keyword_filter
            logger.info(f"Starting Reddit download - Target: {args.target_posts:,} posts")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Resume: {not args.no_resume}")
            logger.info(f"Keyword filtering: {'Enabled' if filter_by_keywords else 'Disabled'}")
            
            if filter_by_keywords and hasattr(config, 'LLM_KEYWORDS') and config.LLM_KEYWORDS:
                logger.info(f"LLM keywords: {', '.join(config.LLM_KEYWORDS)}")
            
            start_time = datetime.now()
            
            # Download posts
            downloaded_file = downloader.download_posts(
                target_posts=args.target_posts,
                output_dir=args.output_dir,
                resume=not args.no_resume,
                filter_by_keywords=filter_by_keywords
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Show final statistics
            stats = downloader.get_download_stats(downloaded_file)
            
            logger.info("Download completed!")
            logger.info(f"Duration: {duration}")
            logger.info(f"Total Posts: {stats.get('total_posts', 0):,}")
            logger.info(f"Unique Subreddits: {stats.get('unique_subreddits', 0)}")
            logger.info(f"Data saved to: {downloaded_file}")
            
            # Save download summary
            summary = {
                'download_time': start_time.isoformat(),
                'completion_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'target_posts': args.target_posts,
                'actual_posts': stats.get('total_posts', 0),
                'unique_subreddits': stats.get('unique_subreddits', 0),
                'output_file': downloaded_file,
                'keyword_filtering': filter_by_keywords,
                'settings': {
                    'subreddits': config.SUBREDDITS,
                    'llm_keywords': getattr(config, 'LLM_KEYWORDS', []),
                    'requests_per_window': RedditDownloader.REQUESTS_PER_WINDOW,
                    'posts_per_request': RedditDownloader.POSTS_PER_REQUEST,
                    'window_duration': RedditDownloader.WINDOW_DURATION
                }
            }
            
            summary_file = os.path.join(args.output_dir, "download_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Download summary saved to: {summary_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 