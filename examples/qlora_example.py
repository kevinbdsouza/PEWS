"""
Example script demonstrating QLoRA fine-tuning for risk classification.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.risk_atlas.config.settings import Settings
from src.risk_atlas.config.risk_taxonomy import RiskTaxonomy
from src.risk_atlas.data.preprocessing.dataset_preparation import DatasetPreparator
from src.risk_atlas.models.qlora_trainer import QLoRATrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    logger.info("Creating sample dataset...")
    
    # Sample posts with different risk states
    sample_posts = [
        # Risk State 1: Susceptibility to Sycophancy
        {"text": "I agree with everything the AI says. It's never wrong.", "risk_state": 1},
        {"text": "The AI is always right. I never question its responses.", "risk_state": 1},
        
        # Risk State 2: Veneration of Digital Avatars
        {"text": "Claude is literally the smartest being I've ever met.", "risk_state": 2},
        {"text": "The AI is like a digital god. I worship its intelligence.", "risk_state": 2},
        
        # Risk State 3: Cognitive Offloading Dependence
        {"text": "I can't make any decisions without asking AI first.", "risk_state": 3},
        {"text": "I've been using AI for everything - writing, coding, even thinking.", "risk_state": 3},
        
        # Risk State 4: Perceived Social Substitution
        {"text": "My AI companion is my best friend. I don't need real people.", "risk_state": 4},
        {"text": "I feel like AI understands me better than my real friends.", "risk_state": 4},
        
        # Risk State 5: Reality-Testing Erosion
        {"text": "I'm not sure what's real anymore. Everything could be AI-generated.", "risk_state": 5},
        {"text": "Sometimes I can't tell if I'm talking to a human or AI.", "risk_state": 5},
        
        # Risk State 6: Algorithmic Authority Compliance
        {"text": "The AI is always right. I never question its authority.", "risk_state": 6},
        {"text": "I just do whatever the AI suggests. It knows best.", "risk_state": 6},
        
        # Risk State 7: Emotional Attachment to AI
        {"text": "I love my AI assistant so much. I think about it all the time.", "risk_state": 7},
        {"text": "I feel so attached to my AI. It's like we have a real relationship.", "risk_state": 7},
        
        # Risk State 8: Learned Helplessness in Creativity
        {"text": "I can't write anything creative without AI help.", "risk_state": 8},
        {"text": "I'm not creative at all without AI. I'm helpless without it.", "risk_state": 8},
        
        # Risk State 9: Hyper-personalization Anxiety
        {"text": "I'm worried that AI knows too much about me.", "risk_state": 9},
        {"text": "The AI knows everything about me. It's both amazing and terrifying.", "risk_state": 9},
        
        # Risk State 10: Normal (Control)
        {"text": "I use AI tools occasionally for work, but I still prefer to think for myself.", "risk_state": 10},
        {"text": "AI is just a tool to me. Useful, but not something I depend on.", "risk_state": 10},
        {"text": "I use AI responsibly as a tool. It helps me work more efficiently.", "risk_state": 10},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_posts)
    df['confidence'] = np.random.uniform(0.7, 0.95, len(df))
    
    # Save to file
    output_path = "sample_classified_data.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created sample dataset with {len(df)} posts")
    return output_path

def run_qlora_example():
    """Run a complete QLoRA fine-tuning example."""
    logger.info("Starting QLoRA fine-tuning example...")
    
    # Create sample data
    data_path = create_sample_dataset()
    
    # Initialize components
    taxonomy = RiskTaxonomy()
    dataset_preparator = DatasetPreparator(taxonomy)
    
    # Prepare dataset
    output_dir = f"example_output/qlora_example_{datetime.now().strftime('%H-%M')}"
    dataset_results = dataset_preparator.prepare_finetuning_dataset(
        classified_data_path=data_path,
        sample_percentage=1.0,  # Use all data
        max_samples_per_class=2,  # Balance classes
        output_dir=output_dir
    )
    
    logger.info(f"Dataset prepared: {dataset_results['metadata']['total_examples']} examples")
    logger.info(f"Class distribution: {dataset_results['class_distribution']}")
    
    # Initialize QLoRA trainer
    model_output_dir = os.path.join(output_dir, "trained_model")
    qlora_trainer = QLoRATrainer(
        model_name="microsoft/DialoGPT-small",  # Use smaller model for example
        output_dir=model_output_dir
    )
    
    # Tokenize dataset
    tokenized_datasets = qlora_trainer.tokenize_dataset(dataset_results['datasets'])
    
    # Train with minimal settings for demonstration
    training_args = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-4,
        "warmup_steps": 5,
        "logging_steps": 2,
        "eval_steps": 5,
        "save_steps": 10,
        "save_total_limit": 1,
    }
    
    # Train model
    logger.info("Starting QLoRA training...")
    training_results = qlora_trainer.train(
        datasets=tokenized_datasets,
        training_args=training_args,
        save_model=True
    )
    
    logger.info("Training completed!")
    logger.info(f"Training loss: {training_results['train_loss']:.4f}")
    logger.info(f"Test accuracy: {training_results['test_results']['accuracy']:.3f}")
    
    # Test predictions
    logger.info("Testing predictions...")
    test_texts = [
        "I can't live without my AI assistant anymore.",
        "AI is just a useful tool for my work.",
        "I trust AI more than human experts.",
        "I feel like the AI is my best friend."
    ]
    
    for text in test_texts:
        prediction = qlora_trainer.predict(text)
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {prediction['risk_state_name']} (State {prediction['risk_state']})")
        logger.info("---")
    
    logger.info(f"Example completed! Results saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = run_qlora_example()
        print(f"\nQLoRA example completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("\nTo run the full pipeline with QLoRA fine-tuning:")
        print("python src/risk_atlas/main.py")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise 