"""
Free version of Risk Atlas pipeline - uses only free APIs (Reddit + Gemini).
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import json
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.risk_atlas.config.settings import Settings
from src.risk_atlas.config.risk_taxonomy import RiskTaxonomy
from src.risk_atlas.data.collectors.reddit_collector import RedditCollector
from src.risk_atlas.data.collectors.reddit_downloader import RedditDownloader
from src.risk_atlas.data.preprocessing.deduplication import SemanticDeduplicator
from src.risk_atlas.data.preprocessing.dataset_preparation import DatasetPreparator
from src.risk_atlas.models.gemini_classifier import GeminiClassifier
from src.risk_atlas.models.qlora_trainer import QLoRATrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_atlas_free.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskAtlasPipelineFree:
    """Free version of Risk Atlas pipeline (Reddit + Gemini only)."""
    
    def __init__(self, config: Settings):
        """
        Initialize the free Risk Atlas pipeline.
        
        Args:
            config: Free configuration settings
        """
        self.config = config
        self.output_dir = f"output/risk_atlas/{datetime.now().strftime('%d-%m-%Y-%H-%M')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components (Reddit only)
        self.taxonomy = RiskTaxonomy()
        self.reddit_collector = RedditCollector(config)
        self.reddit_downloader = RedditDownloader(config)
        self.deduplicator = SemanticDeduplicator(
            model_name=config.EMBEDDING_MODEL,
            cluster_radius=config.DEDUPLICATION_RADIUS
        )
        self.gemini_classifier = GeminiClassifier(config.GEMINI_API_KEY)
        self.dataset_preparator = DatasetPreparator(self.taxonomy)
        self.qlora_trainer = QLoRATrainer(
            model_name=config.QLORA_MODEL_NAME,
            output_dir=os.path.join(self.output_dir, "qlora_model"),
            num_classes=10  # 10 risk states
        )
        
        logger.info(f"Initialized Risk Atlas pipeline. Output directory: {self.output_dir}")
        logger.info("Using Reddit API, local downloads, and Gemini API")
        
    def run_full_pipeline(self, collect_data: bool = True, 
                         deduplicate: bool = True,
                         classify: bool = True,
                         finetune: bool = True,
                         max_posts: Optional[int] = None,
                         use_local_data: bool = False,
                         local_data_path: Optional[str] = None,
                         sample_percentage: float = 0.02,
                         max_samples_per_class: Optional[int] = None) -> Dict:
        """
        Execute complete Risk Atlas pipeline (free version).
        
        Args:
            collect_data: Whether to collect new data
            deduplicate: Whether to perform deduplication
            classify: Whether to perform classification
            max_posts: Maximum number of posts to process (for testing)
            
        Returns:
            Dictionary with pipeline results
        """
        results = {
            'start_time': datetime.now().isoformat(),
            'output_dir': self.output_dir,
            'steps_completed': []
        }
        
        try:
            # Phase 1: Data Collection (Reddit API or local data)
            if collect_data:
                if use_local_data:
                    logger.info("Phase 1: Loading local Reddit data")
                    data_results = self._load_local_reddit_data(local_data_path, max_posts)
                else:
                    logger.info("Phase 1: Collecting Reddit data from API")
                    data_results = self._collect_reddit_data(max_posts)
                results['data_collection'] = data_results
                results['steps_completed'].append('data_collection')
            else:
                # Load existing data
                data_path = os.path.join(self.output_dir, 'raw_data.csv')
                if os.path.exists(data_path):
                    data_results = {'raw_data_path': data_path}
                    logger.info(f"Loaded existing data from {data_path}")
                else:
                    raise FileNotFoundError(f"No existing data found at {data_path}")
            
            # Phase 2: Preprocessing
            if deduplicate:
                logger.info("Phase 2: Preprocessing data...")
                preprocessing_results = self._preprocess_data(data_results['raw_data_path'])
                results['preprocessing'] = preprocessing_results
                results['steps_completed'].append('preprocessing')
            else:
                preprocessing_results = {'deduplicated_data_path': data_results['raw_data_path']}
            
            # Phase 3: Classification (Gemini)
            if classify:
                logger.info("Phase 3: Classifying posts with Gemini")
                classification_results = self._classify_posts(preprocessing_results['deduplicated_data_path'])
                results['classification'] = classification_results
                results['steps_completed'].append('classification')
            
            # Phase 4: Dataset Preparation and QLoRA Fine-tuning
            if finetune and classify:
                logger.info("Phase 4: Preparing dataset and fine-tuning QLoRA model")
                finetuning_results = self._prepare_and_finetune(
                    classification_results['classification_results_path']
                )
                results['finetuning'] = finetuning_results
                results['steps_completed'].append('finetuning')
            
            # Save pipeline results
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'completed'
            
            with open(os.path.join(self.output_dir, 'pipeline_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("pipeline completed successfully!")
            logger.info(f"Total cost: $0")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            
            with open(os.path.join(self.output_dir, 'pipeline_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            raise
    
    def _collect_reddit_data(self, max_posts: Optional[int] = None) -> Dict:
        """Collect data from Reddit only."""
        logger.info("Collecting Reddit posts...")
        
        # Collect posts with keyword filtering enabled
        logger.info("Using LLM keyword filtering for targeted data collection")
        combined_data = self.reddit_collector.collect_posts(
            limit_per_subreddit=200,  # Conservative limit for free tier
            time_filter="month",  # Focus on recent posts
            filter_by_keywords=True  # Enable keyword filtering
        )
        
        # Limit posts if specified
        if max_posts and len(combined_data) > max_posts:
            combined_data = combined_data.sample(n=max_posts, random_state=42)
            logger.info(f"Limited to {max_posts} posts for testing")
        
        # Save raw data
        raw_data_path = os.path.join(self.output_dir, 'raw_data.csv')
        combined_data.to_csv(raw_data_path, index=False)
        
        logger.info(f"Collected {len(combined_data)} Reddit posts with keyword filtering (FREE)")
        
        return {
            'total_posts': len(combined_data),
            'reddit_posts': len(combined_data),
            'twitter_posts': 0,
            'raw_data_path': raw_data_path,
            'subreddits_queried': self.config.SUBREDDITS,
            'data_source': 'reddit_api',
            'keyword_filtering': True
        }
    
    def _load_local_reddit_data(self, local_data_path: Optional[str], 
                               max_posts: Optional[int] = None) -> Dict:
        """Load data from locally downloaded Reddit posts."""
        # Determine data file path
        if local_data_path:
            data_file = local_data_path
        else:
            # Default location
            data_file = "data/reddit_downloads/reddit_posts.csv"
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Local Reddit data not found at {data_file}")
        
        logger.info(f"Loading local Reddit data from {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} posts from local file")
        
        # Limit posts if specified
        if max_posts and len(df) > max_posts:
            df = df.sample(n=max_posts, random_state=42)
            logger.info(f"Limited to {max_posts} posts for testing")
        
        # Save to pipeline output directory
        raw_data_path = os.path.join(self.output_dir, 'raw_data.csv')
        df.to_csv(raw_data_path, index=False)
        
        # Get statistics
        stats = self.reddit_downloader.get_download_stats(data_file)
        
        logger.info(f"Loaded {len(df)} local Reddit posts")
        
        return {
            'total_posts': len(df),
            'reddit_posts': len(df),
            'twitter_posts': 0,
            'raw_data_path': raw_data_path,
            'data_source': 'local_download',
            'source_file': data_file,
            'stats': stats
        }
    
    def _preprocess_data(self, raw_data_path: str) -> Dict:
        """Preprocess and deduplicate data."""
        # Load raw data
        df = pd.read_csv(raw_data_path)
        
        if len(df) == 0:
            logger.warning("No data to preprocess")
            return {
                'original_posts': 0,
                'deduplicated_posts': 0,
                'reduction_ratio': 0,
                'deduplicated_data_path': raw_data_path
            }
        
        # Determine text column
        text_column = 'text'
        if 'text' not in df.columns and 'selftext' in df.columns:
            text_column = 'selftext'
        
        # Deduplicate
        logger.info(f"Deduplicating {len(df)} posts...")
        deduplicated_df = self.deduplicator.deduplicate_corpus(
            df, 
            text_column=text_column,
            save_embeddings=True,
            embeddings_path=os.path.join(self.output_dir, 'embeddings.npy')
        )
        
        # Save deduplicated data
        deduplicated_path = os.path.join(self.output_dir, 'deduplicated_data.csv')
        deduplicated_df.to_csv(deduplicated_path, index=False)
        
        logger.info(f"Deduplication complete: {len(df)} -> {len(deduplicated_df)} posts")
        
        return {
            'original_posts': len(df),
            'deduplicated_posts': len(deduplicated_df),
            'reduction_ratio': len(deduplicated_df) / len(df) if len(df) > 0 else 0,
            'deduplicated_data_path': deduplicated_path,
            'embeddings_path': os.path.join(self.output_dir, 'embeddings.npy')
        }
    
    def _classify_posts(self, data_path: str) -> Dict:
        """Classify posts using Gemini (free tier), accumulating until each class has at least 1000 posts."""
        # Load deduplicated data
        df = pd.read_csv(data_path)
        if len(df) == 0:
            logger.warning("No data to classify")
            return {
                'total_classified': 0,
                'state_distribution': {},
                'classification_results_path': None,
                'analysis_dir': None
            }
        
        text_column = 'text'
        if 'text' not in df.columns and 'selftext' in df.columns:
            text_column = 'selftext'
        
        # Accumulate classified posts until each class has at least 1000
        required_per_class = 1000
        classified_dfs = []
        class_counts = {i: 0 for i in range(1, 11)}
        used_indices = set()
        batch_size = min(self.config.SEED_LABEL_SIZE, 1000)
        rng = np.random.default_rng(42)
        logger.info(f"Classifying batches until each class has at least {required_per_class} posts...")
        
        while min(class_counts.values()) < required_per_class:
            # Sample a batch of unclassified posts
            available_indices = list(set(df.index) - used_indices)
            if not available_indices:
                logger.warning("Ran out of posts before reaching 1000 per class. Stopping early.")
                break
            sample_indices = rng.choice(available_indices, size=min(batch_size, len(available_indices)), replace=False)
            batch_df = df.loc[sample_indices]
            used_indices.update(sample_indices)
            
            # Classify batch
            batch_results = self.gemini_classifier.classify_dataframe(
                batch_df,
                text_column=text_column,
                save_results=False
            )
            classified_dfs.append(batch_results)
            
            # Update class counts
            for state_id, count in batch_results['risk_state'].value_counts().items():
                class_counts[state_id] = class_counts.get(state_id, 0) + count
            logger.info(f"Current class counts: {class_counts}")
        
        # Concatenate all classified batches
        if classified_dfs:
            all_classified = pd.concat(classified_dfs, ignore_index=True)
        else:
            all_classified = pd.DataFrame()
        
        # Save full classified dataset
        classified_path = os.path.join(self.output_dir, 'classification_results.csv')
        all_classified.to_csv(classified_path, index=False)
        
        # Save detailed results
        results_dir = os.path.join(self.output_dir, 'classification_analysis')
        self.gemini_classifier.save_classification_results(all_classified, results_dir)
        
        # Analyze results
        state_counts = all_classified['risk_state'].value_counts().sort_index()
        logger.info("Final classification results:")
        for state_id, count in state_counts.items():
            state_name = self.taxonomy.get_state_by_id(state_id).name
            logger.info(f"  {state_name}: {count} posts")
        
        return {
            'total_classified': len(all_classified),
            'state_distribution': state_counts.to_dict(),
            'classification_results_path': classified_path,
            'analysis_dir': results_dir
        }
    
    def _prepare_and_finetune(self, 
                             classification_results_path: str) -> Dict:
        """Prepare dataset and fine-tune QLoRA model using all classified posts."""
        logger.info("Preparing dataset for QLoRA fine-tuning (using all classified posts)...")
        
        # Prepare dataset (use all classified posts, no sampling or class balancing)
        dataset_dir = os.path.join(self.output_dir, "finetuning_dataset")
        dataset_results = self.dataset_preparator.prepare_finetuning_dataset(
            classified_data_path=classification_results_path,
            sample_percentage=1.0,
            max_samples_per_class=None,
            output_dir=dataset_dir
        )
        
        logger.info(f"Dataset prepared: {dataset_results['metadata']['total_examples']} examples")
        logger.info(f"Class distribution: {dataset_results['class_distribution']}")
        
        # Tokenize dataset
        tokenized_datasets = self.qlora_trainer.tokenize_dataset(dataset_results['datasets'])
        
        # Fine-tune model
        logger.info("Starting QLoRA fine-tuning...")
        training_results = self.qlora_trainer.train(
            datasets=tokenized_datasets,
            save_model=True
        )
        
        logger.info("QLoRA fine-tuning completed!")
        logger.info(f"Test accuracy: {training_results['test_results']['accuracy']:.3f}")
        
        return {
            'dataset_path': dataset_dir,
            'dataset_metadata': dataset_results['metadata'],
            'class_distribution': dataset_results['class_distribution'],
            'training_results': training_results,
            'model_path': self.qlora_trainer.output_dir
        }
    
    def generate_report(self) -> str:
        """Generate a summary report of the pipeline results."""
        report_path = os.path.join(self.output_dir, 'pipeline_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Risk Atlas Pipeline Report (FREE VERSION)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Cost: $0\n\n")
            
            # Load pipeline results
            results_path = os.path.join(self.output_dir, 'pipeline_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as rf:
                    results = json.load(rf)
                
                f.write("Pipeline Status: {}\n".format(results.get('status', 'unknown')))
                f.write("Steps Completed: {}\n\n".format(', '.join(results.get('steps_completed', []))))
                
                # Data collection results
                if 'data_collection' in results:
                    data = results['data_collection']
                    data_source = data.get('data_source', 'unknown')
                    f.write(f"Data Collection ({data_source.upper()}):\n")
                    f.write(f"  Total Posts: {data.get('total_posts', 0)}\n")
                    f.write(f"  Reddit Posts: {data.get('reddit_posts', 0)}\n")
                    f.write(f"  Twitter Posts: {data.get('twitter_posts', 0)} (not used in free version)\n")
                    
                    if data_source == 'reddit_api':
                        f.write(f"  Subreddits: {', '.join(data.get('subreddits_queried', []))}\n")
                        if data.get('keyword_filtering'):
                            f.write("  Keyword Filtering: Enabled\n")
                    elif data_source == 'local_download':
                        f.write(f"  Source File: {data.get('source_file', 'unknown')}\n")
                        if 'stats' in data:
                            stats = data['stats']
                            f.write(f"  Unique Subreddits: {stats.get('unique_subreddits', 0)}\n")
                            f.write(f"  Date Range: {stats.get('date_range', {}).get('earliest', 'N/A')} to {stats.get('date_range', {}).get('latest', 'N/A')}\n")
                    f.write("\n")
                
                # Preprocessing results
                if 'preprocessing' in results:
                    preproc = results['preprocessing']
                    f.write("Preprocessing:\n")
                    f.write(f"  Original Posts: {preproc.get('original_posts', 0)}\n")
                    f.write(f"  Deduplicated Posts: {preproc.get('deduplicated_posts', 0)}\n")
                    f.write(f"  Reduction Ratio: {preproc.get('reduction_ratio', 0):.2%}\n\n")
                
                # Classification results
                if 'classification' in results:
                    classif = results['classification']
                    f.write("Classification (Gemini FREE tier):\n")
                    f.write(f"  Total Classified: {classif.get('total_classified', 0)}\n\n")
                    
                    f.write("Risk State Distribution:\n")
                    state_dist = classif.get('state_distribution', {})
                    for state_id, count in sorted(state_dist.items()):
                        state_name = self.taxonomy.get_state_by_id(int(state_id)).name
                        f.write(f"  {state_name}: {count} posts\n")
                
                # Fine-tuning results
                if 'finetuning' in results:
                    finetune = results['finetuning']
                    f.write("\nQLoRA Fine-tuning Results:\n")
                    f.write("=" * 30 + "\n")
                    
                    # Dataset info
                    dataset_meta = finetune.get('dataset_metadata', {})
                    f.write(f"Dataset Size: {dataset_meta.get('total_examples', 0)} examples\n")
                    f.write(f"Train Size: {dataset_meta.get('train_size', 0)} examples\n")
                    f.write(f"Validation Size: {dataset_meta.get('val_size', 0)} examples\n")
                    f.write(f"Test Size: {dataset_meta.get('test_size', 0)} examples\n")
                    f.write(f"Sample Percentage: {dataset_meta.get('sample_percentage', 0):.1%}\n")
                    
                    # Class distribution
                    f.write("\nClass Distribution:\n")
                    class_dist = finetune.get('class_distribution', {})
                    for class_name, count in class_dist.items():
                        f.write(f"  {class_name}: {count} examples\n")
                    
                    # Training results
                    training_results = finetune.get('training_results', {})
                    f.write(f"\nTraining Loss: {training_results.get('train_loss', 0):.4f}\n")
                    f.write(f"Training Runtime: {training_results.get('train_runtime', 0):.1f} seconds\n")
                    
                    # Test results
                    test_results = training_results.get('test_results', {})
                    f.write(f"Test Accuracy: {test_results.get('accuracy', 0):.3f}\n")
                    f.write(f"Test Loss: {test_results.get('eval_loss', 0):.4f}\n")
                    
                    f.write(f"\nModel saved to: {finetune.get('model_path', 'N/A')}\n")
        
        logger.info(f"Generated report: {report_path}")
        return report_path

def main():
    """Main entry point for the Risk Atlas pipeline."""
    parser = argparse.ArgumentParser(description='Risk Atlas Pipeline')
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-deduplication', action='store_true',
                       help='Skip deduplication step')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip classification step')
    parser.add_argument('--skip-finetuning', action='store_true',
                       help='Skip QLoRA fine-tuning step')
    parser.add_argument('--max-posts', type=int, default=1000,
                       help='Maximum number of posts to process (default: 1000)')
    parser.add_argument('--sample-percentage', type=float, default=0.02,
                       help='Percentage of classified posts to use for fine-tuning (default: 0.02)')
    parser.add_argument('--max-samples-per-class', type=int,
                       help='Maximum samples per risk state class for balancing')
    parser.add_argument('--use-local-data', action='store_true',
                       help='Use locally downloaded Reddit data instead of API')
    parser.add_argument('--local-data-path', type=str,
                       help='Path to local Reddit data file (default: data/reddit_downloads/reddit_posts.csv)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Settings()
    
    # Validate configuration
    if not config.validate_config():
        logger.error("Invalid configuration. Please check your API keys.")
        logger.info("Get FREE API keys:")
        logger.info("  Reddit: https://www.reddit.com/prefs/apps")
        logger.info("  Gemini: https://makersuite.google.com/app/apikey")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = RiskAtlasPipelineFree(config)
        
        results = pipeline.run_full_pipeline(
            collect_data=not args.skip_data_collection,
            deduplicate=not args.skip_deduplication,
            classify=not args.skip_classification,
            finetune=not args.skip_finetuning,
            max_posts=args.max_posts,
            use_local_data=args.use_local_data,
            local_data_path=args.local_data_path,
            sample_percentage=args.sample_percentage,
            max_samples_per_class=args.max_samples_per_class
        )
        
        # Generate report
        report_path = pipeline.generate_report()
        
        logger.info("pipeline completed successfully!")
        logger.info(f"Results saved to: {pipeline.output_dir}")
        logger.info(f"Report generated: {report_path}")

        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 