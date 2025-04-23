import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import logging
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'compare_models_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_comparison')

class ModelComparator:
    def __init__(self):
        """Initialize the model comparator"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.plots_dir = 'model_comparison_plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.standard_model = None
        self.standard_tokenizer = None
        self.combined_model = None
        self.combined_tokenizer = None
        
        self.test_data = None
        self.comparison_results = None
        
        logger.info(f"Model comparator initialized. Plots will be saved to {self.plots_dir}")
    
    def load_models(self, standard_model_dir=None, combined_model_dir=None):
        """Load the standard and combined models"""
        # Find the most recent model directories if not specified
        models_dir = 'models'
        
        if standard_model_dir is None:
            # Find most recent standard model
            standard_models = [d for d in os.listdir(models_dir) 
                               if os.path.isdir(os.path.join(models_dir, d)) and 
                               'standard_sentiment' in d]
            
            if not standard_models:
                logger.error("No standard sentiment model found")
                return False
            
            # Sort by date (newest first)
            standard_model_dir = os.path.join(models_dir, sorted(standard_models, reverse=True)[0])
        
        if combined_model_dir is None:
            # Find most recent combined model
            combined_models = [d for d in os.listdir(models_dir) 
                               if os.path.isdir(os.path.join(models_dir, d)) and 
                               'combined_sentiment' in d]
            
            if not combined_models:
                logger.error("No combined sentiment model found")
                return False
            
            # Sort by date (newest first)
            combined_model_dir = os.path.join(models_dir, sorted(combined_models, reverse=True)[0])
        
        try:
            # Load standard model
            logger.info(f"Loading standard model from {standard_model_dir}")
            self.standard_tokenizer = AutoTokenizer.from_pretrained(standard_model_dir)
            self.standard_model = AutoModelForSequenceClassification.from_pretrained(standard_model_dir)
            self.standard_model.to(self.device)
            
            # Load combined model
            logger.info(f"Loading combined model from {combined_model_dir}")
            self.combined_tokenizer = AutoTokenizer.from_pretrained(combined_model_dir)
            self.combined_model = AutoModelForSequenceClassification.from_pretrained(combined_model_dir)
            self.combined_model.to(self.device)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def load_test_data(self, test_file=None):
        """Load the test data for model comparison"""
        # Find the test data file if not specified
        if test_file is None:
            test_file = os.path.join('data', 'test_data.csv')
            combined_test_file = os.path.join('data', 'combined_test_data.csv')
            
            # Prefer combined test data if available
            if os.path.exists(combined_test_file):
                test_file = combined_test_file
        
        try:
            self.test_data = pd.read_csv(test_file)
            logger.info(f"Loaded {len(self.test_data)} test samples from {test_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return False
    
    def predict_with_models(self, text, tokenizer, model):
        """Make a prediction with a specific model"""
        # Tokenize the input
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # Get confidence for each class
        confidence = probabilities.cpu().numpy()[0]
        
        return prediction, confidence
    
    def sentiment_to_text(self, sentiment_id):
        """Convert sentiment ID to text label"""
        sentiment_mapping = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        return sentiment_mapping.get(sentiment_id, 'unknown')
    
    def compare_models(self):
        """Compare standard and combined models on test data"""
        if self.standard_model is None or self.combined_model is None:
            logger.error("Models not loaded")
            return False
        
        if self.test_data is None:
            logger.error("Test data not loaded")
            return False
        
        logger.info("Comparing standard and combined models")
        
        # Use the column containing review text
        text_column = 'clean_text'
        if 'train_text' in self.test_data.columns:
            text_column = 'train_text'
        elif 'enhanced_text' in self.test_data.columns:
            text_column = 'enhanced_text'
        
        # Ground truth labels
        true_labels = self.test_data['label'].tolist()
        true_label_texts = [self.sentiment_to_text(label) for label in true_labels]
        
        # Lists to store results
        standard_predictions = []
        standard_confidences = []
        combined_predictions = []
        combined_confidences = []
        
        # Make predictions with both models
        for i, text in enumerate(self.test_data[text_column]):
            # Standard model prediction
            std_pred, std_conf = self.predict_with_models(text, self.standard_tokenizer, self.standard_model)
            standard_predictions.append(std_pred)
            standard_confidences.append(std_conf)
            
            # Combined model prediction
            comb_pred, comb_conf = self.predict_with_models(text, self.combined_tokenizer, self.combined_model)
            combined_predictions.append(comb_pred)
            combined_confidences.append(comb_conf)
            
            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(self.test_data)} test samples")
        
        # Convert predictions to text labels
        standard_prediction_texts = [self.sentiment_to_text(pred) for pred in standard_predictions]
        combined_prediction_texts = [self.sentiment_to_text(pred) for pred in combined_predictions]
        
        # Create comparison results
        self.comparison_results = pd.DataFrame({
            'text': self.test_data[text_column],
            'true_label': true_labels,
            'true_label_text': true_label_texts,
            'standard_prediction': standard_predictions,
            'standard_prediction_text': standard_prediction_texts,
            'combined_prediction': combined_predictions,
            'combined_prediction_text': combined_prediction_texts
        })
        
        # Add metadata
        if 'difficulty' in self.test_data.columns:
            self.comparison_results['difficulty'] = self.test_data['difficulty']
        
        if 'would_take_again_bool' in self.test_data.columns:
            self.comparison_results['would_take_again'] = self.test_data['would_take_again_bool']
        
        if 'department' in self.test_data.columns:
            self.comparison_results['department'] = self.test_data['department']
        
        # Calculate agreement and disagreement
        self.comparison_results['models_agree'] = (
            self.comparison_results['standard_prediction'] == 
            self.comparison_results['combined_prediction']
        )
        
        self.comparison_results['standard_correct'] = (
            self.comparison_results['standard_prediction'] == 
            self.comparison_results['true_label']
        )
        
        self.comparison_results['combined_correct'] = (
            self.comparison_results['combined_prediction'] == 
            self.comparison_results['true_label']
        )
        
        # Save comparison results
        results_path = os.path.join(self.plots_dir, 'model_comparison_results.csv')
        self.comparison_results.to_csv(results_path, index=False)
        logger.info(f"Saved model comparison results to {results_path}")
        
        # Calculate agreement percentage
        agreement_pct = (self.comparison_results['models_agree'].sum() / len(self.comparison_results)) * 100
        logger.info(f"Models agree on {agreement_pct:.2f}% of test samples")
        
        # Calculate accuracy for each model
        standard_accuracy = (self.comparison_results['standard_correct'].sum() / len(self.comparison_results)) * 100
        combined_accuracy = (self.comparison_results['combined_correct'].sum() / len(self.comparison_results)) * 100
        
        logger.info(f"Standard model accuracy: {standard_accuracy:.2f}%")
        logger.info(f"Combined model accuracy: {combined_accuracy:.2f}%")
        
        return True
    
    def analyze_disagreements(self):
        """Analyze cases where the models disagree"""
        if self.comparison_results is None:
            logger.error("No comparison results available")
            return
        
        # Get disagreement cases
        disagreements = self.comparison_results[~self.comparison_results['models_agree']]
        disagreement_count = len(disagreements)
        
        if disagreement_count == 0:
            logger.info("No disagreements found between models")
            return
        
        logger.info(f"Analyzing {disagreement_count} disagreement cases")
        
        # Identify which model was correct in disagreement cases
        disagreements['standard_was_correct'] = disagreements['standard_correct'] & ~disagreements['combined_correct']
        disagreements['combined_was_correct'] = disagreements['combined_correct'] & ~disagreements['standard_correct']
        disagreements['both_wrong'] = ~disagreements['standard_correct'] & ~disagreements['combined_correct']
        
        # Calculate counts
        standard_correct_count = disagreements['standard_was_correct'].sum()
        combined_correct_count = disagreements['combined_was_correct'].sum()
        both_wrong_count = disagreements['both_wrong'].sum()
        
        # Calculate percentages
        standard_correct_pct = (standard_correct_count / disagreement_count) * 100
        combined_correct_pct = (combined_correct_count / disagreement_count) * 100
        both_wrong_pct = (both_wrong_count / disagreement_count) * 100
        
        logger.info(f"In disagreement cases:")
        logger.info(f"  Standard model was correct: {standard_correct_count} ({standard_correct_pct:.2f}%)")
        logger.info(f"  Combined model was correct: {combined_correct_count} ({combined_correct_pct:.2f}%)")
        logger.info(f"  Both models were wrong: {both_wrong_count} ({both_wrong_pct:.2f}%)")
        
        # Plot disagreement resolution
        plt.figure(figsize=(10, 6))
        disagreement_resolution = pd.Series({
            'Standard Correct': standard_correct_count,
            'Combined Correct': combined_correct_count,
            'Both Wrong': both_wrong_count
        })
        
        disagreement_resolution.plot(kind='bar')
        plt.title('Resolution of Model Disagreements')
        plt.xlabel('Resolution')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'disagreement_resolution.png')
        plt.savefig(plot_path)
        logger.info(f"Saved disagreement resolution plot to {plot_path}")
        
        # Save interesting disagreement examples
        interesting_examples = disagreements.sample(min(10, disagreement_count))
        examples_path = os.path.join(self.plots_dir, 'interesting_disagreements.csv')
        interesting_examples.to_csv(examples_path, index=False)
        logger.info(f"Saved interesting disagreement examples to {examples_path}")
        
        # Analyze difficulty in disagreement cases (if available)
        if 'difficulty' in disagreements.columns:
            # Calculate average difficulty for each resolution type
            std_correct_difficulty = disagreements[disagreements['standard_was_correct']]['difficulty'].mean()
            comb_correct_difficulty = disagreements[disagreements['combined_was_correct']]['difficulty'].mean()
            
            logger.info(f"Average difficulty when standard model is correct: {std_correct_difficulty:.2f}")
            logger.info(f"Average difficulty when combined model is correct: {comb_correct_difficulty:.2f}")
            
            # Plot difficulty distribution by correct model
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='standard_was_correct', y='difficulty', data=disagreements)
            plt.title('Difficulty Distribution by Correct Model')
            plt.xlabel('Standard Model Correct')
            plt.ylabel('Difficulty Rating')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'disagreement_difficulty.png')
            plt.savefig(plot_path)
            logger.info(f"Saved disagreement difficulty plot to {plot_path}")
    
    def summarize_comparison(self):
        """Generate a summary of the model comparison"""
        if self.comparison_results is None:
            logger.error("No comparison results available")
            return
        
        logger.info("Generating model comparison summary")
        
        # Overall accuracy
        standard_accuracy = (self.comparison_results['standard_correct'].sum() / len(self.comparison_results)) * 100
        combined_accuracy = (self.comparison_results['combined_correct'].sum() / len(self.comparison_results)) * 100
        
        # Agreement rate
        agreement_rate = (self.comparison_results['models_agree'].sum() / len(self.comparison_results)) * 100
        
        # Create summary dictionary
        summary = {
            'test_samples': len(self.comparison_results),
            'standard_model_accuracy': standard_accuracy,
            'combined_model_accuracy': combined_accuracy,
            'agreement_rate': agreement_rate,
            'disagreement_count': len(self.comparison_results[~self.comparison_results['models_agree']]),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Get class-wise accuracy for each model
        for model_type in ['standard', 'combined']:
            for label in [0, 1, 2]:  # negative, neutral, positive
                label_samples = self.comparison_results[self.comparison_results['true_label'] == label]
                
                if len(label_samples) > 0:
                    accuracy = (label_samples[f'{model_type}_correct'].sum() / len(label_samples)) * 100
                    summary[f'{model_type}_accuracy_class_{label}'] = accuracy
        
        # Save summary to JSON
        summary_path = os.path.join(self.plots_dir, 'model_comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Saved model comparison summary to {summary_path}")
        
        # Print summary to console
        print("\n========== MODEL COMPARISON SUMMARY ==========")
        print(f"Test samples: {summary['test_samples']}")
        print(f"Standard model accuracy: {summary['standard_model_accuracy']:.2f}%")
        print(f"Combined model accuracy: {summary['combined_model_accuracy']:.2f}%")
        print(f"Model agreement rate: {summary['agreement_rate']:.2f}%")
        print(f"Number of disagreements: {summary['disagreement_count']}")
        print("==============================================\n")
        
        return summary
    
    def run_comparison(self):
        """Run the complete model comparison analysis"""
        self.compare_models()
        self.analyze_disagreements()
        summary = self.summarize_comparison()
        return summary

def main():
    comparator = ModelComparator()
    
    # Load models and test data
    if comparator.load_models() and comparator.load_test_data():
        comparator.run_comparison()
    else:
        logger.error("Failed to load models or test data. Comparison aborted.")

if __name__ == "__main__":
    main()