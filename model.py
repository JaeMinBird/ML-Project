import os
import logging
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
# Updated import for AdamW from torch.optim instead of transformers
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'model_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sentiment_model')

class ReviewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=128):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
        
class SentimentClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=3):
        """Initialize the DistilBERT sentiment classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        logger.info(f"Initialized {model_name} model with {num_labels} output classes")
    
    def _create_data_loader(self, df, batch_size=16, text_col='clean_text'):
        """Create a DataLoader from a dataframe"""
        # Use enhanced text if available, otherwise fall back to clean_text or train_text
        if 'train_text' in df.columns:
            text_col = 'train_text'
        elif text_col not in df.columns and 'enhanced_text' in df.columns:
            text_col = 'enhanced_text'
            
        logger.info(f"Creating data loader using {text_col} column")
        
        dataset = ReviewsDataset(
            reviews=df[text_col].to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=self.tokenizer
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2
        )
    
    def train(self, train_df, val_df=None, epochs=4, batch_size=16, learning_rate=2e-5):
        """Train the model on the given data"""
        logger.info(f"Starting training on {len(train_df)} samples")
        
        # Prepare data loaders
        train_data_loader = self._create_data_loader(train_df, batch_size=batch_size)
        
        val_data_loader = None
        if val_df is not None:
            val_data_loader = self._create_data_loader(val_df, batch_size=batch_size)
            logger.info(f"Will validate on {len(val_df)} samples")
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_data_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_progress = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for batch in epoch_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_progress.set_postfix(loss=loss.item())
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / len(train_data_loader)
            train_losses.append(avg_train_loss)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_data_loader is not None:
                val_loss, val_accuracy, val_f1 = self.evaluate(val_data_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Plot training progress
        self._plot_training_progress(train_losses, val_losses, val_accuracies)
        
        return train_losses, val_losses, val_accuracies
    
    def evaluate(self, data_loader):
        """Evaluate the model on the given data"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                _, preds = torch.max(outputs.logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Determine unique classes in the actual targets and predictions
        unique_classes = sorted(set(np.unique(all_targets)) | set(np.unique(all_predictions)))
        
        # Full list of possible target names
        all_target_names = ['negative', 'neutral', 'positive']
        
        # Filter to include only the target names for classes that are actually present
        target_names = [all_target_names[i] for i in unique_classes]
        
        try:
            # Print classification report with appropriate target names
            report = classification_report(
                all_targets, 
                all_predictions, 
                labels=unique_classes,
                target_names=target_names
            )
            logger.info(f"Classification Report:\n{report}")
        except Exception as e:
            logger.error(f"Could not generate classification report: {e}")
            # Fallback: generate report without target_names
            report = classification_report(all_targets, all_predictions)
            logger.info(f"Classification Report (without target names):\n{report}")
        
        return avg_loss, accuracy, f1
    
    def _plot_training_progress(self, train_losses, val_losses=None, val_accuracies=None):
        """Plot the training progress"""
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot validation accuracy
        if val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
        
        # Save plot
        os.makedirs('logs', exist_ok=True)
        plot_path = os.path.join('logs', f'training_progress_{datetime.now().strftime("%Y%m%d")}.png')
        plt.savefig(plot_path)
        logger.info(f"Training progress plot saved to {plot_path}")
    
    def predict(self, texts):
        """Make predictions on new texts"""
        self.model.eval()
        
        # Tokenize inputs
        encoded_texts = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
        
        # Map numerical predictions to sentiment labels
        sentiment_mapping = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
        return [sentiment_mapping[pred.item()] for pred in preds]
    
    def save_model(self, save_dir="models", model_type="standard"):
        """Save the model and tokenizer"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_dir, f'professor_reviews_{model_type}_sentiment_{timestamp}')
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        logger.info(f"{model_type.capitalize()} model saved to {model_dir}")
        return model_dir
    
    @classmethod
    def load_model(cls, model_dir):
        """Load a saved model and tokenizer"""
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        classifier = cls(model_name=None)
        classifier.model = model
        classifier.tokenizer = tokenizer
        
        classifier.model.to(classifier.device)
        
        logger.info(f"Model loaded from {model_dir}")
        return classifier


def fine_tune_model(train_path, test_path=None, epochs=4, batch_size=16, learning_rate=2e-5, model_type="standard"):
    """Fine-tune a DistilBERT model on the RateMyProfessors data"""
    # Load training data
    try:
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded training data from {train_path}: {len(train_df)} samples")
        
        test_df = None
        if test_path:
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded test data from {test_path}: {len(test_df)} samples")
        
        # Initialize and train model
        classifier = SentimentClassifier()
        
        train_losses, val_losses, val_accuracies = classifier.train(
            train_df=train_df,
            val_df=test_df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save the model
        model_dir = classifier.save_model(model_type=model_type)
        
        # Final evaluation on test set
        if test_df is not None:
            test_data_loader = classifier._create_data_loader(test_df, batch_size=batch_size)
            test_loss, test_accuracy, test_f1 = classifier.evaluate(test_data_loader)
            
            logger.info(f"Final Test Results for {model_type} model:")
            logger.info(f"Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
            
            # Save metrics to a file
            with open(os.path.join('logs', f'{model_type}_test_metrics_{datetime.now().strftime("%Y%m%d")}.txt'), 'w') as f:
                f.write(f"Test Loss: {test_loss:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                f.write(f"Test F1 Score: {test_f1:.4f}\n")
        
        return model_dir
            
    except Exception as e:
        logger.error(f"Error during model fine-tuning: {e}")
        return None


def fine_tune_combined_model(train_path, test_path=None, epochs=4, batch_size=16, learning_rate=2e-5):
    """Fine-tune a DistilBERT model using the combined sentiment score (rating + difficulty)"""
    return fine_tune_model(train_path, test_path, epochs, batch_size, learning_rate, model_type="combined")


def main():
    """Main function to run the model fine-tuning process"""
    # Find train and test files
    data_dir = 'data'
    train_file = os.path.join(data_dir, 'train_data.csv')
    test_file = os.path.join(data_dir, 'test_data.csv')
    
    combined_train_file = os.path.join(data_dir, 'combined_train_data.csv')
    combined_test_file = os.path.join(data_dir, 'combined_test_data.csv')
    
    if not os.path.exists(train_file):
        logger.error(f"Training data file not found: {train_file}")
        return
    
    logger.info("Starting model fine-tuning process")
    
    # First, train the standard sentiment model
    model_dir = fine_tune_model(train_file, test_file)
    
    if model_dir:
        logger.info(f"Standard model fine-tuning completed successfully. Model saved to {model_dir}")
    else:
        logger.error("Standard model fine-tuning failed")
    
    # Then, if combined data exists, train the combined sentiment model
    if os.path.exists(combined_train_file):
        logger.info("Starting combined model fine-tuning process")
        combined_model_dir = fine_tune_combined_model(combined_train_file, combined_test_file)
        
        if combined_model_dir:
            logger.info(f"Combined model fine-tuning completed successfully. Model saved to {combined_model_dir}")
        else:
            logger.error("Combined model fine-tuning failed")
    else:
        logger.info("No combined training data found. Skipping combined model training.")


if __name__ == "__main__":
    main()