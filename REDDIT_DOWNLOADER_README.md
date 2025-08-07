# Reddit Data Downloader

This module provides a robust Reddit data downloader that respects Reddit API rate limits and can download large datasets for the Risk Atlas project.

## Features

- **Rate Limiting**: Respects Reddit API limits (1000 requests per 10 minutes, 100 posts per request)
- **Resumable Downloads**: Can resume interrupted downloads from where they left off
- **Progress Tracking**: Shows download progress and saves intermediate results
- **Configurable Target**: Download any number of posts (default: 1 million)
- **Multiple Sort Types**: Downloads from hot, top, new, and rising posts
- **Time Filters**: Supports various time filters for top posts
- **Duplicate Prevention**: Automatically removes duplicate posts

## Usage

### 1. Download Reddit Data

Use the standalone downloader script:

```bash
# Download 1 million posts (default)
python src/risk_atlas/download_reddit_data.py

# Download custom number of posts
python src/risk_atlas/download_reddit_data.py --target-posts 500000

# Specify custom output directory
python src/risk_atlas/download_reddit_data.py --output-dir data/my_reddit_data

# Start fresh (don't resume from existing download)
python src/risk_atlas/download_reddit_data.py --no-resume

# Check statistics of existing download
python src/risk_atlas/download_reddit_data.py --stats-only
```

### 2. Use Local Data in Pipeline

After downloading data, you can use it in the main pipeline:

```bash
# Use local data instead of Reddit API
python src/risk_atlas/main.py --use-local-data

# Specify custom local data path
python src/risk_atlas/main.py --use-local-data --local-data-path data/my_reddit_data/reddit_posts.csv

# Combine with other options
python src/risk_atlas/main.py --use-local-data --max-posts 10000 --skip-deduplication
```

## Configuration

### Environment Variables

Make sure you have your Reddit API credentials in your `.env` file:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

### Subreddits

The downloader uses the subreddits defined in `config/settings.py`:

```python
SUBREDDITS = [
    "ArtificialIntelligence", "ChatGPT", "MentalHealth", 
    "MachineLearning", "AI", "OpenAI", "Anthropic", "ClaudeAI",
    "GeminiAI", "AIethics", "psychology", "depression", "anxiety",
    "GPT", "Claude", "AIcompanion", "AIrelationship"
]
```

## Rate Limiting Details

The downloader implements strict rate limiting:

- **1000 requests per 10-minute window**
- **100 posts per request**
- **Automatic sleep when limits are reached**
- **Progress tracking across rate limit windows**

## Output Files

The downloader creates several files:

- `reddit_posts.csv`: Main data file with all downloaded posts
- `download_progress.json`: Progress tracking for resuming downloads
- `download_summary.json`: Summary of the download process
- `reddit_download.log`: Log file with detailed information

## Data Format

Each downloaded post contains:

```python
{
    'platform': 'reddit',
    'subreddit': 'subreddit_name',
    'post_id': 'unique_post_id',
    'title': 'post_title',
    'text': 'post_content',
    'score': upvote_count,
    'num_comments': comment_count,
    'created_utc': datetime,
    'author': 'author_name',
    'url': 'post_url',
    'is_self': boolean,
    'over_18': boolean,
    'spoiler': boolean,
    'stickied': boolean,
    'upvote_ratio': float,
    'downloaded_at': datetime
}
```

## Performance Considerations

### Download Speed

With rate limiting, the theoretical maximum download speed is:
- 1000 requests × 100 posts = 100,000 posts per 10 minutes
- 100,000 posts per 600 seconds = ~167 posts per second

### Memory Usage

The downloader saves intermediate results to avoid memory issues:
- Posts are saved in batches
- Memory is cleared after each batch
- Large datasets are handled efficiently

### Storage Requirements

Estimated storage for 1 million posts:
- CSV file: ~200-500 MB (depending on post content)
- Progress files: ~1-2 MB
- Total: ~250-600 MB

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: The downloader should handle these automatically, but if you see them, the downloader will pause and resume.

2. **Interrupted Downloads**: Use the `--resume` flag (default) to continue from where you left off.

3. **Memory Issues**: The downloader saves intermediate results, so memory usage should be minimal.

4. **API Key Issues**: Make sure your Reddit API credentials are correct and have the necessary permissions.

### Log Files

Check the log files for detailed information:
- `reddit_download.log`: Download process logs
- `risk_atlas_free.log`: Main pipeline logs

## Integration with Main Pipeline

The main pipeline can now switch between two data sources:

1. **Reddit API** (original): Real-time data collection with limited posts
2. **Local Downloads**: Large datasets downloaded separately

This allows for:
- Faster pipeline runs with pre-downloaded data
- Larger datasets for analysis
- Reduced API usage
- More consistent data for research

## Example Workflow

1. **Download large dataset**:
   ```bash
   python src/risk_atlas/download_reddit_data.py --target-posts 1000000
   ```

2. **Run pipeline with local data**:
   ```bash
   python src/risk_atlas/main.py --use-local-data --max-posts 50000
   ```

3. **Check statistics**:
   ```bash
   python src/risk_atlas/download_reddit_data.py --stats-only
   ```

This approach gives you the flexibility to download large datasets once and use them for multiple pipeline runs, while still maintaining the ability to collect fresh data when needed. 