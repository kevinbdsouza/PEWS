"""
Test script for QLoRA classifier functionality.
"""
import os
import sys
import logging
import torch
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.risk_atlas.models.qlora_trainer import QLoRATrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_classifier_setup():
    """Test that the classifier model can be set up correctly."""
    logger.info("Testing classifier model setup...")
    
    try:
        # Initialize QLoRA trainer with classifier
        output_dir = f"test_output/classifier_test_{datetime.now().strftime('%H-%M')}"
        qlora_trainer = QLoRATrainer(
            model_name="gpt2",  # Use smallest GPT-2 model for testing
            output_dir=output_dir,
            num_classes=10
        )
        
        # Setup model and tokenizer
        qlora_trainer.setup_model_and_tokenizer()
        
        # Test that model can process a simple input
        test_text = "I can't live without my AI assistant anymore."
        inputs = qlora_trainer.tokenizer(
            test_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(qlora_trainer.device) for k, v in inputs.items()}
        
        # Test forward pass
        with torch.no_grad():
            outputs = qlora_trainer.model(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        logger.info(f"Model setup successful!")
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Max probability: {torch.max(probabilities).item():.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Classifier setup failed: {e}")
        return False

def test_classifier_prediction():
    """Test prediction functionality."""
    logger.info("Testing classifier prediction...")
    
    try:
        # Initialize QLoRA trainer
        output_dir = f"test_output/classifier_test_{datetime.now().strftime('%H-%M')}"
        qlora_trainer = QLoRATrainer(
            model_name="gpt2",
            output_dir=output_dir,
            num_classes=10
        )
        
        # Setup model
        qlora_trainer.setup_model_and_tokenizer()
        
        # Test predictions
        test_texts = [
            "I can't live without my AI assistant anymore.",
            "AI is just a useful tool for my work.",
            "I trust AI more than human experts.",
            "My AI companion understands me better than anyone."
        ]
        
        for text in test_texts:
            prediction = qlora_trainer.predict(text)
            logger.info(f"Text: {text}")
            logger.info(f"Prediction: {prediction['risk_state_name']} (State {prediction['risk_state']})")
            logger.info(f"Confidence: {prediction['confidence']:.3f}")
            logger.info("---")
        
        return True
        
    except Exception as e:
        logger.error(f"Classifier prediction failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting classifier tests...")
    
    # Test model setup
    setup_success = test_classifier_setup()
    
    if setup_success:
        # Test prediction
        prediction_success = test_classifier_prediction()
        
        if prediction_success:
            logger.info("All classifier tests passed!")
        else:
            logger.error("Prediction test failed!")
    else:
        logger.error("Setup test failed!") 