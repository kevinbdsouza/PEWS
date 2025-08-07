"""
QLoRA fine-tuning for risk state classification with classification head (GPT-2 compatible).
"""
import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)

class ClassificationHead(nn.Module):
    """Classification head for risk state classification."""
    
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hidden_states):
        # Use the last hidden state for classification
        pooled_output = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class QLoRAClassifier(nn.Module):
    """QLoRA model with classification head for risk state classification."""
    
    def __init__(self, base_model, num_classes: int = 10):
        super().__init__()
        self.base_model = base_model
        self.classification_head = ClassificationHead(
            hidden_size=base_model.config.hidden_size,
            num_classes=num_classes
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Apply classification head
        logits = self.classification_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }

class QLoRATrainer:
    """QLoRA fine-tuning trainer for risk state classification with classification head."""
    
    def __init__(self, 
                 model_name: str = "gpt2-medium",
                 output_dir: str = "models/qlora_risk_classifier",
                 device: str = "auto",
                 num_classes: int = 10):
        """
        Initialize QLoRA trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            device: Device to use for training
            num_classes: Number of risk state classes
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        self.num_classes = num_classes
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"Initialized QLoRA classifier trainer with model: {model_name}")
    
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer for QLoRA fine-tuning."""
        logger.info("Setting up model and tokenizer...")
        
        # Load base model (no quantization for GPT-2)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float32  # Use float32 for GPT-2
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA for GPT-2 architecture
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]  # GPT-2 specific modules
        )
        
        # Apply LoRA to base model
        base_model = get_peft_model(base_model, lora_config)
        
        # Create classifier model
        self.model = QLoRAClassifier(base_model, num_classes=self.num_classes)
        
        # Move model to device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        self.model = self.model.to(self.device)
        
        # Print trainable parameters
        trainable_params = 0
        all_params = 0
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
        logger.info("Model and tokenizer setup complete")
    
    def tokenize_dataset(self, datasets: DatasetDict, max_length: int = 512) -> DatasetDict:
        """Tokenize the dataset for classification training."""
        logger.info("Tokenizing dataset for classification...")
        
        def tokenize_function(examples):
            # For classification, we just need the input text and labels
            texts = examples['input']  # Just the post text
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Add labels
            tokenized["labels"] = examples['label']  # Direct class labels
            
            return tokenized
        
        # Apply tokenization to all splits
        tokenized_datasets = {}
        for split_name, dataset in datasets.items():
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
        
        return DatasetDict(tokenized_datasets)
    
    def train(self, 
              datasets: DatasetDict,
              training_args: Optional[Dict] = None,
              save_model: bool = True) -> Dict:
        """
        Train the model using QLoRA with classification head.
        
        Args:
            datasets: Tokenized datasets
            training_args: Training arguments
            save_model: Whether to save the trained model
            
        Returns:
            Training results
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Default training arguments (optimized for classification)
        default_args = {
            "output_dir": self.output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 500,
            "save_total_limit": 3,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_accuracy",
            "greater_is_better": True,
            "report_to": None,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "dataloader_pin_memory": False,
            "fp16": True,
            "tf32": True,
        }
        
        # Update with provided arguments
        if training_args:
            default_args.update(training_args)
        
        # Create training arguments
        training_args = TrainingArguments(**default_args)
        
        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )
        
        # Train the model
        logger.info("Starting QLoRA classification training...")
        train_result = self.trainer.train()
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Evaluate on test set
        test_results = self.evaluate(datasets["test"])
        
        # Compile results
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "test_results": test_results,
            "model_path": self.output_dir if save_model else None
        }
        
        # Save training results
        with open(os.path.join(self.output_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        return results
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "predictions": predictions.tolist(),
            "labels": labels.tolist()
        }
    
    def evaluate(self, test_dataset: Dataset) -> Dict:
        """Evaluate the model on test dataset."""
        logger.info("Evaluating model on test set...")
        
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Run evaluation
        eval_results = self.trainer.evaluate(test_dataset)
        
        # Generate predictions for detailed analysis
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(min(len(test_dataset), 100)):  # Sample for analysis
                example = test_dataset[i]
                inputs = {k: torch.tensor(v).unsqueeze(0).to(self.model.device) 
                         for k, v in example.items() if k != "labels"}
                
                outputs = self.model(**inputs)
                logits = outputs['logits']
                
                # Get predicted class
                predicted_class = torch.argmax(logits[0]).item()
                predictions.append(predicted_class)
                
                # Get true label
                true_label = example["labels"]
                true_labels.append(true_label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Generate classification report
        class_names = [f"Risk_State_{i}" for i in range(1, self.num_classes + 1)]
        report = classification_report(
            true_labels, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        results = {
            "eval_loss": eval_results["eval_loss"],
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": predictions[:10],  # Save first 10 for inspection
            "true_labels": true_labels[:10]
        }
        
        logger.info(f"Test accuracy: {accuracy:.3f}")
        return results
    
    def save_model(self):
        """Save the fine-tuned model and weights."""
        logger.info(f"Saving model to {self.output_dir}")
        
        # Save the model
        self.trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save LoRA config
        self.model.base_model.save_pretrained(self.output_dir)
        
        # Save classification head weights separately
        torch.save(
            self.model.classification_head.state_dict(),
            os.path.join(self.output_dir, "classification_head.pt")
        )
        
        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "hidden_size": self.model.base_model.config.hidden_size,
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["c_attn", "c_proj", "c_fc"]
            }
        }
        
        with open(os.path.join(self.output_dir, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Model and weights saved successfully")
    
    def load_model(self, model_path: str):
        """Load a fine-tuned model."""
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model with LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float32
        )
        
        # Load model configuration
        config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            self.num_classes = config["num_classes"]
        else:
            self.num_classes = 10  # Default
        
        # Create classifier model
        self.model = QLoRAClassifier(base_model, num_classes=self.num_classes)
        
        # Load classification head weights
        head_path = os.path.join(model_path, "classification_head.pt")
        if os.path.exists(head_path):
            self.model.classification_head.load_state_dict(
                torch.load(head_path, map_location=self.device)
            )
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> Dict:
        """Make a prediction on a single text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be loaded before making predictions")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map class ID to risk state
        risk_states = {
            0: "01_Information_Overload_Anxiety",
            1: "02_Digital_Identity_Fragmentation", 
            2: "03_Cognitive_Offloading_Dependence",
            3: "04_Perceived_Social_Substitution",
            4: "05_Reality-Testing_Erosion",
            5: "06_Algorithmic_Authority_Compliance",
            6: "07_Emotional_Attachment_to_AI",
            7: "08_Learned_Helplessness_in_Creativity",
            8: "09_Hyper-personalization_Anxiety",
            9: "10_None_of_the_Above"
        }
        
        return {
            "risk_state": predicted_class + 1,  # Convert to 1-based indexing
            "risk_state_name": risk_states.get(predicted_class, "Unknown"),
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        } 