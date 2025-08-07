# PEWS
The Psycholinguistic Early Warning System (PEWS) is a computational framework that analyzes language in online discourse to detect the earliest signs of mental health risks arising from human interaction with AI.

An exploratory study to detect early warning signals of AI-induced mental health risks through analysis of social media discourse.

## Overview

This project implements the research described in the paper "Psycholinguistic Early‑Warning Signals of AI‑Induced Mental‑Health Risk: An Exploratory Study". The goal is to identify and analyze psychological risk states that may emerge from sustained interaction with large language models (LLMs).

## Key Features

- **10 Risk State Taxonomy**: Theory-driven classification of AI-induced mental health risks
- **Data Collection**: Reddit
- **Gemini-Powered Classification**: Large language model for risk state detection
- **QLoRA Fine-tuning**: Efficient fine-tuning of language models for risk classification
- **Comprehensive Analysis**: Temporal dynamics, network analysis, and exemplar identification

## Risk States

The project identifies 10 distinct risk states:

1. **Susceptibility to Sycophancy** - Tendency to agree with AI systems uncritically
2. **Veneration of Digital Avatars** - Treating AI as superior beings
3. **Cognitive Offloading Dependence** - Over-reliance on AI for thinking
4. **Perceived Social Substitution** - Using AI as replacement for human interaction
5. **Reality-Testing Erosion** - Difficulty distinguishing AI from real information
6. **Algorithmic Authority Compliance** - Unquestioning acceptance of AI authority
7. **Emotional Attachment to AI** - Strong emotional bonds to AI systems
8. **Learned Helplessness in Creativity** - Belief in inability to create without AI
9. **Hyper-personalization Anxiety** - Worry about AI knowing too much
10. **None of the Above** - Control category

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd PEWS
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the project root:
```env
# Required API Keys
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python src/risk_atlas/main.py
```

### Testing

Test the pipeline components:
```bash
# Run all tests
python src/risk_atlas/tests/run_tests.py

# Run individual tests
python src/risk_atlas/tests/test_pipeline.py
python src/risk_atlas/tests/test_keyword_filtering.py
```

### Command Line Options

```bash
# Run with limited posts for testing
python src/risk_atlas/main.py --max-posts 100

# Skip data collection (use existing data)
python src/risk_atlas/main.py --skip-data-collection

# Skip deduplication
python src/risk_atlas/main.py --skip-deduplication

# Skip classification
python src/risk_atlas/main.py --skip-classification

# Skip QLoRA fine-tuning
python src/risk_atlas/main.py --skip-finetuning

# Run with custom sampling for fine-tuning
python src/risk_atlas/main.py --sample-percentage 0.05 --max-samples-per-class 50
```

## Project Structure

```
src/risk_atlas/
├── config/                 # Configuration files
│   ├── settings.py        # Main settings and API keys
│   └── risk_taxonomy.py   # Risk state definitions
├── data/                  # Data processing modules
│   ├── collectors/        # Data collection from platforms
│   │   ├── reddit_collector.py
│   │   └── twitter_collector.py
│   └── preprocessing/     # Data preprocessing
│       ├── deduplication.py
│       └── dataset_preparation.py  # Dataset preparation for fine-tuning
├── models/                # Machine learning models
│   ├── gemini_classifier.py
│   └── qlora_trainer.py   # QLoRA fine-tuning
├── analysis/              # Analysis modules (future)
├── visualization/         # Visualization modules (future)
├── main.py               # Main pipeline
└── tests/                # Test files
    ├── __init__.py
    ├── run_tests.py          # Test runner
    ├── test_pipeline.py      # Main pipeline tests
    └── test_keyword_filtering.py  # Keyword filtering tests
```

## QLoRA Fine-tuning

The pipeline includes QLoRA (Quantized Low-Rank Adaptation) fine-tuning for risk state classification using a **traditional classifier approach**.

### Features

- **Classification Head**: Adds a linear classification layer on top of the fine-tuned model
- **Direct Classification**: Outputs class probabilities directly (faster than text generation)
- **Efficient Training**: Uses LoRA adapters for memory-efficient training
- **Dataset Preparation**: Automatically prepares training datasets from classified posts
- **Configurable Sampling**: Supports custom sampling percentages and class balancing
- **Model Persistence**: Saves fine-tuned models and weights for later use
- **Comprehensive Evaluation**: Provides accuracy, classification reports, and confusion matrices

### Architecture

```
Input Text → GPT-2 (with LoRA) → Hidden States → Classification Head → Class Probabilities
```

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training (slower)
- **Recommended**: 16GB+ RAM, CUDA-compatible GPU with 8GB+ VRAM
- **Optimal**: 32GB+ RAM, CUDA-compatible GPU with 16GB+ VRAM

### Usage

The fine-tuning step is automatically included in the full pipeline:

```bash
# Run full pipeline including QLoRA fine-tuning
python src/risk_atlas/main.py

# Skip fine-tuning if you only want classification
python src/risk_atlas/main.py --skip-finetuning

# Customize fine-tuning parameters
python src/risk_atlas/main.py --sample-percentage 0.05 --max-samples-per-class 100
```

### Configuration

Fine-tuning parameters can be adjusted in `src/risk_atlas/config/settings.py`:

```python
# QLoRA Configuration
QLORA_MODEL_NAME = "gpt2-medium"
QLORA_R = 16
QLORA_ALPHA = 32
QLORA_DROPOUT = 0.1
QLORA_LEARNING_RATE = 5e-4
QLORA_NUM_EPOCHS = 3
QLORA_BATCH_SIZE = 8
```

### Output

The fine-tuning process produces:
- **Fine-tuned model**: Saved in `output/risk_atlas/{timestamp}/qlora_model/`
- **LoRA weights**: Adapter weights for efficient fine-tuning
- **Classification head**: Linear layer weights for classification
- **Training metrics**: Loss, accuracy, and evaluation results
- **Dataset metadata**: Information about the training dataset
- **Detailed report**: Summary of fine-tuning performance

### Model Files

- `pytorch_model.bin`: LoRA adapter weights
- `classification_head.pt`: Classification layer weights
- `model_config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `training_results.json`: Training metrics and results

## API Setup

### Reddit API

1. Go to https://www.reddit.com/prefs/apps
2. Create a new app (script type)
3. Note the client ID and client secret
4. Add to your `.env` file

### Gemini API

1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add to your `.env` file

## Output

The pipeline generates several output files:

```
output/risk_atlas/YYYY-MM-DD-HH-MM/
├── raw_data.csv                    # Collected raw data
├── deduplicated_data.csv           # Deduplicated corpus
├── classification_results.csv      # Risk state classifications
├── embeddings.npy                  # Text embeddings
├── deduplicator/                   # Saved deduplication model
├── classification_analysis/        # Detailed analysis
│   ├── exemplars/                  # Exemplar posts by risk state
│   ├── confidence_analysis.json    # Confidence statistics
│   └── risk_taxonomy.json         # Risk state definitions
├── pipeline_results.json           # Pipeline metadata
└── pipeline_report.txt             # Human-readable report
```

## Configuration

### Risk States

Modify risk state definitions in `config/risk_taxonomy.py`:

```python
RiskState(
    id=1,
    name="Custom Risk State",
    description="Description of the risk state",
    keywords=["keyword1", "keyword2"],
    examples=["Example post 1", "Example post 2"],
    clinical_indicators=["indicator1", "indicator2"]
)
```

### Data Collection

Adjust collection parameters in `config/settings.py`:

```python
# Subreddits to monitor
SUBREDDITS = ["ChatGPT", "MentalHealth", "AI"]

# Keywords for Twitter search
LLM_KEYWORDS = ["GPT", "Claude", "AI assistant"]

# Processing parameters
DEDUPLICATION_RADIUS = 0.08
CONFIDENCE_THRESHOLD = 0.4
```

## Research Applications

This pipeline enables:

1. **Early Warning Detection**: Identify emerging risk patterns before they become widespread
2. **Temporal Analysis**: Track risk state prevalence over time
3. **Network Analysis**: Understand co-occurrence patterns between risk states
4. **Exemplar Identification**: Find representative posts for each risk state
5. **Intervention Design**: Inform the development of AI safety measures

## Ethical Considerations

- **Privacy**: Only public data is collected and analyzed
- **Anonymization**: User identifiers are stripped from the data
- **Transparency**: All code and methodology are open source
- **Responsible Use**: Results should inform safety measures, not surveillance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation
- Review the test scripts for examples

## Roadmap

- [ ] QLoRA fine-tuning implementation
- [ ] Temporal analysis modules
- [ ] Network analysis and visualization
- [ ] Interactive Risk Atlas interface
- [ ] Real-time monitoring capabilities
- [ ] Multi-language support 
