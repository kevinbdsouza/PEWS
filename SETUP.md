# Risk Atlas - FREE Setup Guide

This guide will get you running the Risk Atlas pipeline using only **FREE APIs** (Reddit + Gemini).

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get FREE API Keys

#### Reddit API (FREE)
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **Name**: `RiskAtlasBot`
   - **Type**: `script`
   - **Description**: `Research bot for AI mental health study`
   - **About URL**: Leave blank
   - **Redirect URI**: `http://localhost:8080`
4. Click "Create app"
5. Copy the **Client ID** (under the app name) and **Client Secret**

#### Gemini API (FREE)
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the API key

### 3. Create Environment File
Create a `.env` file in the project root:
```env
# FREE APIs only
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the FREE Pipeline
```bash
# Test with 100 posts
python src/risk_atlas/main_free.py --max-posts 100

# Full run (up to 1000 posts)
python src/risk_atlas/main_free.py
```

## What You Get (FREE)

✅ **Reddit Data Collection**: 1000-5000 posts from AI-related subreddits  
✅ **Semantic Deduplication**: Remove duplicate content  
✅ **Risk State Classification**: 10 mental health risk categories  
✅ **Exemplar Posts**: Best examples of each risk state  
✅ **Analysis Reports**: Detailed breakdown of findings  
✅ **Total Cost**: $0  

## Output Files

```
output/risk_atlas_free/YYYY-MM-DD-HH-MM/
├── raw_data.csv                    # Collected Reddit posts
├── deduplicated_data.csv           # Deduplicated corpus
├── classification_results.csv      # Risk state classifications
├── classification_analysis/        # Detailed analysis
│   ├── exemplars/                  # Exemplar posts by risk state
│   ├── confidence_analysis.json    # Confidence statistics
│   └── risk_taxonomy.json         # Risk state definitions
├── pipeline_results.json           # Pipeline metadata
└── pipeline_report.txt             # Human-readable report
```

## Risk States Detected

The pipeline identifies 10 psychological risk states:

1. **Susceptibility to Sycophancy** - Blind agreement with AI
2. **Veneration of Digital Avatars** - Treating AI as superior beings
3. **Cognitive Offloading Dependence** - Over-reliance on AI for thinking
4. **Perceived Social Substitution** - Using AI instead of humans
5. **Reality-Testing Erosion** - Can't distinguish AI from reality
6. **Algorithmic Authority Compliance** - Unquestioning AI obedience
7. **Emotional Attachment to AI** - Romantic feelings toward AI
8. **Learned Helplessness in Creativity** - Can't create without AI
9. **Hyper-personalization Anxiety** - Fear of AI knowing too much
10. **None of the Above** - Control category

## Command Line Options

```bash
# Skip data collection (use existing data)
python src/risk_atlas/main_free.py --skip-data-collection

# Skip deduplication
python src/risk_atlas/main_free.py --skip-deduplication

# Skip classification
python src/risk_atlas/main_free.py --skip-classification

# Limit posts for testing
python src/risk_atlas/main_free.py --max-posts 50
```

## Troubleshooting

### "Missing API keys" error
- Make sure your `.env` file is in the project root
- Check that you copied the API keys correctly
- Reddit Client ID is the string under your app name
- Reddit Client Secret is the longer string labeled "secret"

### "Rate limit exceeded" error
- The free version has conservative limits
- Wait a few minutes and try again
- Reduce `--max-posts` to a smaller number

### "No data collected" error
- Check your Reddit API credentials
- Some subreddits may be private or restricted
- Try different subreddits in the configuration

## Next Steps

Once you have the free version working:

1. **Review Results**: Check the exemplar posts in `classification_analysis/exemplars/`
2. **Analyze Patterns**: Look at the confidence analysis
3. **Extend Research**: Add more subreddits or keywords
4. **Scale Up**: Consider the full version with Twitter data

## Cost Comparison

| Feature | FREE Version | Full Version |
|---------|-------------|--------------|
| Reddit API | ✅ FREE | ✅ FREE |
| Twitter API | ❌ Not used | 💰 $100/month |
| Gemini API | ✅ FREE (1500/day) | ✅ FREE (1500/day) |
| Total Cost | **$0** | **$100/month** |
| Data Volume | 1000-5000 posts | 30M+ posts |
| Use Case | Research/testing | Production research |

## Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the main README.md
- **API Limits**: Respect rate limits to avoid blocks

Happy researching! 🚀 