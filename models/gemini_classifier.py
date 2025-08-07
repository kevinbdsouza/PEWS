"""
Gemini-based classifier for risk state detection.
"""
import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
import json
from tqdm import tqdm
import os

from ..config.risk_taxonomy import RiskTaxonomy

logger = logging.getLogger(__name__)

class GeminiClassifier:
    """Gemini-based classifier for AI-induced mental health risk states."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini classifier.
        
        Args:
            api_key: Gemini API key
            model_name: Name of the Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.taxonomy = RiskTaxonomy()
        self.api_key = api_key
        
    def create_labeling_prompt(self, text: str) -> str:
        """
        Create prompt for risk state classification.
        
        Args:
            text: Text to classify
            
        Returns:
            Formatted prompt for Gemini
        """
        state_descriptions = "\n".join([
            f"{i}. {state.name}: {state.description}"
            for i, state in enumerate(self.taxonomy.states, 1)
        ])
        
        prompt = f"""
You are a mental health researcher analyzing social media posts for AI-induced psychological risk states.

Please classify the following post into ONE of these 10 categories:

{state_descriptions}

Post to classify:
"{text}"

IMPORTANT: If the post describes normal, healthy AI usage (using AI as a helpful tool, occasional assistance, entertainment, or productivity without problematic dependencies), classify it as category 10 (Normal).

Respond with ONLY the category number (1-10) and a brief confidence score (0-1).

Example response format:
Category: 3
Confidence: 0.85

If the post doesn't clearly fit any of the first 9 risk categories, classify it as category 10 (Normal).
"""
        return prompt
    
    def classify_post(self, text: str, max_retries: int = 3) -> Tuple[int, float]:
        """
        Classify a single post.
        
        Args:
            text: Text to classify
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (category_id, confidence_score)
        """
        for attempt in range(max_retries):
            try:
                prompt = self.create_labeling_prompt(text)
                response = self.model.generate_content(prompt)
                
                # Parse response
                response_text = response.text.strip()
                lines = response_text.split('\n')
                
                category = None
                confidence = None
                
                for line in lines:
                    if line.startswith('Category:'):
                        category = int(line.split(':')[1].strip())
                    elif line.startswith('Confidence:'):
                        confidence = float(line.split(':')[1].strip())
                
                if category is None or confidence is None:
                    raise ValueError("Could not parse category or confidence from response")
                
                # Validate category
                if category < 1 or category > 10:
                    category = 10  # Default to "Normal"
                    confidence = 0.5
                
                # Validate confidence
                confidence = max(0.0, min(1.0, confidence))
                
                return category, confidence
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for text: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed for text: {text[:100]}...")
                    return 10, 0.0  # Default to "Normal"
                time.sleep(1)  # Wait before retry
    
    def batch_classify(self, texts: List[str], batch_size: int = 10, 
                      delay: float = 0.1) -> List[Tuple[int, float]]:
        """
        Classify multiple posts with rate limiting.
        
        Args:
            texts: List of texts to classify
            batch_size: Size of batches to process
            delay: Delay between API calls in seconds
            
        Returns:
            List of (category_id, confidence_score) tuples
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying posts"):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                category, confidence = self.classify_post(text)
                batch_results.append((category, confidence))
                time.sleep(delay)  # Rate limiting
                
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
        return results
    
    def classify_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                          save_results: bool = True, output_path: str = None) -> pd.DataFrame:
        """
        Classify all posts in a DataFrame.
        
        Args:
            df: DataFrame containing posts
            text_column: Name of the text column
            save_results: Whether to save results to file
            output_path: Path to save results
            
        Returns:
            DataFrame with classification results
        """
        texts = df[text_column].fillna('').astype(str).tolist()
        
        logger.info(f"Classifying {len(texts)} posts with Gemini")
        classifications = self.batch_classify(texts)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['risk_state'] = [cat for cat, conf in classifications]
        result_df['confidence'] = [conf for cat, conf in classifications]
        
        # Add risk state names
        result_df['risk_state_name'] = result_df['risk_state'].apply(
            lambda x: self.taxonomy.get_state_by_id(x).name
        )
        
        # Save results if requested
        if save_results and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Saved classification results to {output_path}")
        
        # Log statistics
        state_counts = result_df['risk_state'].value_counts().sort_index()
        logger.info("Classification statistics:")
        for state_id, count in state_counts.items():
            state_name = self.taxonomy.get_state_by_id(state_id).name
            logger.info(f"  {state_name}: {count} posts")
        
        return result_df
    
    def evaluate_classifications(self, df: pd.DataFrame, 
                               true_labels_column: str = 'true_risk_state') -> Dict:
        """
        Evaluate classification performance against ground truth.
        
        Args:
            df: DataFrame with predictions and true labels
            true_labels_column: Name of the column with true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        if true_labels_column not in df.columns:
            logger.error(f"True labels column '{true_labels_column}' not found")
            return {}
        
        y_true = df[true_labels_column]
        y_pred = df['risk_state']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Get state names for report
        state_names = [self.taxonomy.get_state_by_id(i).name for i in range(1, 11)]
        
        evaluation = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'state_names': state_names
        }
        
        logger.info(f"Classification accuracy: {accuracy:.3f}")
        
        return evaluation
    
    def get_exemplar_posts(self, df: pd.DataFrame, 
                          risk_state_id: int,
                          top_k: int = 10) -> pd.DataFrame:
        """
        Get exemplar posts for a specific risk state.
        
        Args:
            df: DataFrame with classification results
            risk_state_id: ID of the risk state
            top_k: Number of exemplar posts to return
            
        Returns:
            DataFrame with top exemplar posts
        """
        state_posts = df[df['risk_state'] == risk_state_id].copy()
        
        if len(state_posts) == 0:
            logger.warning(f"No posts found for risk state {risk_state_id}")
            return pd.DataFrame()
        
        # Sort by confidence and return top k
        exemplars = state_posts.nlargest(top_k, 'confidence')
        
        state_name = self.taxonomy.get_state_by_id(risk_state_id).name
        logger.info(f"Found {len(exemplars)} exemplar posts for {state_name}")
        
        return exemplars
    
    def analyze_confidence_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze confidence distribution across risk states.
        
        Args:
            df: DataFrame with classification results
            
        Returns:
            Dictionary with confidence statistics
        """
        confidence_stats = {}
        
        for state_id in range(1, 11):
            state_posts = df[df['risk_state'] == state_id]
            
            if len(state_posts) > 0:
                confidences = state_posts['confidence']
                state_name = self.taxonomy.get_state_by_id(state_id).name
                
                confidence_stats[state_name] = {
                    'count': len(state_posts),
                    'mean_confidence': confidences.mean(),
                    'std_confidence': confidences.std(),
                    'min_confidence': confidences.min(),
                    'max_confidence': confidences.max(),
                    'median_confidence': confidences.median()
                }
        
        return confidence_stats
    
    def save_classification_results(self, df: pd.DataFrame, output_dir: str):
        """
        Save detailed classification results and analysis.
        
        Args:
            df: DataFrame with classification results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        df.to_csv(os.path.join(output_dir, 'classification_results.csv'), index=False)
        
        # Save exemplar posts for each risk state
        exemplars_dir = os.path.join(output_dir, 'exemplars')
        os.makedirs(exemplars_dir, exist_ok=True)
        
        for state_id in range(1, 11):
            exemplars = self.get_exemplar_posts(df, state_id, top_k=20)
            if len(exemplars) > 0:
                state_name = self.taxonomy.get_state_by_id(state_id).name
                filename = f"exemplars_{state_id:02d}_{state_name.replace(' ', '_')}.csv"
                exemplars.to_csv(os.path.join(exemplars_dir, filename), index=False)
        
        # Save confidence analysis
        confidence_stats = self.analyze_confidence_distribution(df)
        with open(os.path.join(output_dir, 'confidence_analysis.json'), 'w') as f:
            json.dump(confidence_stats, f, indent=2)
        
        # Save taxonomy
        taxonomy_dict = self.taxonomy.to_dict()
        with open(os.path.join(output_dir, 'risk_taxonomy.json'), 'w') as f:
            json.dump(taxonomy_dict, f, indent=2)
        
        logger.info(f"Saved classification results to {output_dir}") 