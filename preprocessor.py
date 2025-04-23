import pandas as pd
import os
import logging
from datetime import datetime
import re
from transformers import AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'preprocessor_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('preprocessor')

class TextPreprocessor:
    def __init__(self, model_name="distilbert-base-uncased"):
        """Initialize the preprocessor with a specific tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Initialized preprocessor with {model_name} tokenizer")
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def assign_sentiment_label(self, rating):
        """
        Assign sentiment label based on rating
        1-2: Negative
        3: Neutral
        4-5: Positive
        """
        if not isinstance(rating, (int, float)):
            return 'neutral'  # Default if rating is not numeric

        try:
            rating_val = float(rating)
            if rating_val <= 2:
                return 'negative'
            elif rating_val <= 3:
                return 'neutral'
            else:
                return 'positive'
        except (ValueError, TypeError):
            return 'neutral'  # Default if rating cannot be converted
    
    def parse_tags(self, tags_str):
        """Extract tag information from comma-separated string"""
        if not isinstance(tags_str, str):
            return []
        
        # Split by comma and strip whitespace
        return [tag.strip() for tag in tags_str.split(',') if tag.strip()]
    
    def extract_tag_features(self, tags_list):
        """
        Extract common features from tags.
        Returns a dictionary of boolean features indicating presence of certain qualities.
        """
        tag_features = {
            'is_helpful': False,
            'is_tough': False,
            'gives_feedback': False,
            'lots_of_homework': False,
            'clear_grading': False,
            'caring': False,
            'respected': False,
            'inspirational': False,
            'skip_class': False,
            'tough_grader': False,
            'get_ready_to_read': False
        }
        
        # Convert all tags to lowercase for case-insensitive matching
        lower_tags = [tag.lower() for tag in tags_list]
        
        # Check for key phrases in the tags
        if any(tag in lower_tags for tag in ['helpful', 'super helpful']):
            tag_features['is_helpful'] = True
            
        if any(tag in lower_tags for tag in ['tough', 'difficult', 'challenging']):
            tag_features['is_tough'] = True
            
        if any('feedback' in tag for tag in lower_tags):
            tag_features['gives_feedback'] = True
            
        if any(tag in lower_tags for tag in ['lots of homework', 'heavy workload']):
            tag_features['lots_of_homework'] = True
            
        if any(tag in lower_tags for tag in ['clear grading', 'fair grading']):
            tag_features['clear_grading'] = True
            
        if any(tag in lower_tags for tag in ['caring', 'compassionate', 'understanding']):
            tag_features['caring'] = True
            
        if any(tag in lower_tags for tag in ['respected', 'respected professor']):
            tag_features['respected'] = True
            
        if any(tag in lower_tags for tag in ['inspirational', 'inspiring']):
            tag_features['inspirational'] = True
            
        if any(tag in lower_tags for tag in ['skip class', 'skip', 'attendance not mandatory']):
            tag_features['skip_class'] = True
            
        if any(tag in lower_tags for tag in ['tough grader', 'hard grader']):
            tag_features['tough_grader'] = True
            
        if any(tag in lower_tags for tag in ['get ready to read', 'lots of reading']):
            tag_features['get_ready_to_read'] = True
            
        return tag_features
    
    def tokenize_text(self, text, max_length=128):
        """Tokenize text using the BERT tokenizer"""
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
    
    def process_reviews_data(self, csv_path):
        """
        Process the scraped reviews data:
        1. Clean text
        2. Assign sentiment labels based on ratings
        3. Extract additional features from difficulty, would_take_again, and tags
        4. Save processed data
        """
        logger.info(f"Processing reviews from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} reviews from CSV")
            
            # Clean the text
            df['clean_text'] = df['text'].apply(self.clean_text)
            
            # Filter out empty reviews
            df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
            logger.info(f"After removing empty reviews: {len(df)} reviews")
            
            # Assign sentiment labels
            df['sentiment'] = df['rating'].apply(self.assign_sentiment_label)
            
            # Process the tags
            if 'tags' in df.columns:
                # Parse tags to list
                df['tags_list'] = df['tags'].apply(self.parse_tags)
                
                # Extract feature flags from tags
                tag_features_df = pd.DataFrame([
                    self.extract_tag_features(tags) for tags in df['tags_list']
                ])
                
                # Combine with main dataframe
                df = pd.concat([df, tag_features_df], axis=1)
                
                # Count the number of positive tags
                positive_tag_cols = ['is_helpful', 'gives_feedback', 'clear_grading', 
                                    'caring', 'respected', 'inspirational']
                df['positive_tag_count'] = df[positive_tag_cols].sum(axis=1)
                
                # Count the number of negative tags
                negative_tag_cols = ['is_tough', 'lots_of_homework', 'skip_class', 
                                     'tough_grader', 'get_ready_to_read']
                df['negative_tag_count'] = df[negative_tag_cols].sum(axis=1)
                
                # Combine tags with text for enhanced sentiment analysis
                df['enhanced_text'] = df.apply(
                    lambda row: row['clean_text'] + ' ' + ' '.join(row['tags_list']), 
                    axis=1
                )
            else:
                # If no tags column, use clean_text as enhanced_text
                df['enhanced_text'] = df['clean_text']
                
            # Process difficulty rating if available
            if 'difficulty' in df.columns:
                # Normalize difficulty to 0-1 range
                df['normalized_difficulty'] = df['difficulty'].apply(
                    lambda x: float(x)/5.0 if pd.notnull(x) and isinstance(x, (int, float)) else 0.5
                )
                
                # Create difficulty categories
                df['difficulty_level'] = df['difficulty'].apply(
                    lambda x: 'easy' if x <= 2 else ('moderate' if x <= 3.5 else 'hard')
                    if pd.notnull(x) and isinstance(x, (int, float)) else 'unknown'
                )
            
            # Process would_take_again if available
            if 'would_take_again' in df.columns:
                # Convert to boolean (1.0 = Yes, 0.0 = No, None = unknown)
                df['would_take_again_bool'] = df['would_take_again'].apply(
                    lambda x: True if x == 1.0 else (False if x == 0.0 else None)
                )
            
            # Create a combined score using multiple factors
            # This can be used for more nuanced sentiment analysis
            if all(col in df.columns for col in ['rating', 'difficulty']):
                # Formula: rating * (1 - normalized_difficulty/2)
                # This gives higher scores to high-rated, lower difficulty professors
                df['combined_score'] = df.apply(
                    lambda row: row['rating'] * (1 - row.get('normalized_difficulty', 0.5)/2)
                    if pd.notnull(row['rating']) else None, 
                    axis=1
                )
                
                # Create combined sentiment using the combined score
                df['combined_sentiment'] = df['combined_score'].apply(
                    lambda x: 'negative' if x <= 2 else ('neutral' if x <= 3 else 'positive')
                    if pd.notnull(x) else 'neutral'
                )
            
            # Save the processed dataframe
            processed_path = csv_path.replace('.csv', '_processed.csv')
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed data to {processed_path}")
            
            # Count sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
            
            # Generate enhanced statistics if we have the additional data
            if 'combined_sentiment' in df.columns:
                combined_sentiment_counts = df['combined_sentiment'].value_counts()
                logger.info(f"Combined sentiment distribution: {combined_sentiment_counts.to_dict()}")
                
                # Create a comparison between regular sentiment and combined sentiment
                sentiment_comparison = pd.crosstab(df['sentiment'], df['combined_sentiment'])
                logger.info(f"Sentiment vs Combined Sentiment Comparison:\n{sentiment_comparison}")
                
            if 'difficulty_level' in df.columns:
                # Check how difficulty correlates with sentiment
                difficulty_sentiment = pd.crosstab(df['difficulty_level'], df['sentiment'])
                logger.info(f"Difficulty Level vs Sentiment Distribution:\n{difficulty_sentiment}")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Error processing reviews data: {e}")
            return None
            
    def prepare_for_training(self, processed_csv_path, train_test_split=0.8, random_state=42):
        """
        Prepare the processed data for model training:
        1. Load processed CSV
        2. Create numerical labels
        3. Split into train/test sets
        4. Save prepared data
        """
        try:
            df = pd.read_csv(processed_csv_path)
            logger.info(f"Preparing {len(df)} processed reviews for training")
            
            # Create numerical labels
            sentiment_mapping = {
                'negative': 0,
                'neutral': 1,
                'positive': 2
            }
            df['label'] = df['sentiment'].map(sentiment_mapping)
            
            # Use enhanced_text instead of clean_text if available for better training
            if 'enhanced_text' in df.columns:
                logger.info("Using enhanced text (with tags) for training")
                df['train_text'] = df['enhanced_text']
            else:
                df['train_text'] = df['clean_text']
            
            # Add a combined model training option if available
            if 'combined_sentiment' in df.columns:
                logger.info("Preparing data for combined sentiment model training")
                df['combined_label'] = df['combined_sentiment'].map(sentiment_mapping)
                
                # Create a separate training file for the combined model
                combined_train_df = df.copy()
                combined_train_df['label'] = combined_train_df['combined_label']
                
                # Shuffle the data
                combined_train_df = combined_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
                
                # Split into train and test sets
                train_size = int(len(combined_train_df) * train_test_split)
                combined_train_set = combined_train_df.iloc[:train_size]
                combined_test_set = combined_train_df.iloc[train_size:]
                
                logger.info(f"Split combined model data into {len(combined_train_set)} training and {len(combined_test_set)} testing samples")
                
                # Save combined model train and test sets
                base_dir = os.path.dirname(processed_csv_path)
                combined_train_path = os.path.join(base_dir, 'combined_train_data.csv')
                combined_test_path = os.path.join(base_dir, 'combined_test_data.csv')
                
                combined_train_set.to_csv(combined_train_path, index=False)
                combined_test_set.to_csv(combined_test_path, index=False)
                
                logger.info(f"Saved combined training data to {combined_train_path}")
                logger.info(f"Saved combined testing data to {combined_test_path}")
            
            # Shuffle the data for standard sentiment model
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Split into train and test sets
            train_size = int(len(df) * train_test_split)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            logger.info(f"Split data into {len(train_df)} training and {len(test_df)} testing samples")
            
            # Save train and test sets
            base_dir = os.path.dirname(processed_csv_path)
            train_path = os.path.join(base_dir, 'train_data.csv')
            test_path = os.path.join(base_dir, 'test_data.csv')
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            return train_path, test_path
            
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            return None, None


def main():
    # Find the most recent reviews CSV file
    data_dir = 'data'
    review_files = [f for f in os.listdir(data_dir) if f.startswith('penn_state_reviews_') and f.endswith('.csv')]
    
    if not review_files:
        logger.error("No review files found to process")
        return
    
    # Sort by creation time (newest first)
    latest_file = sorted(review_files, reverse=True)[0]
    csv_path = os.path.join(data_dir, latest_file)
    
    preprocessor = TextPreprocessor()
    processed_path = preprocessor.process_reviews_data(csv_path)
    
    if processed_path:
        preprocessor.prepare_for_training(processed_path)
    

if __name__ == "__main__":
    main()