#!/usr/bin/env python3
"""
RateMyProfessors Sentiment Analysis Pipeline
===========================================

This program scrapes professor reviews from RateMyProfessors,
processes the data, labels sentiments based on ratings,
and fine-tunes a DistilBERT model for sentiment analysis.

Usage:
    python main.py --scrape          # Run just the scraping process
    python main.py --process         # Process the most recent scraped data
    python main.py --train           # Train on the most recent processed data
    python main.py --run-all         # Run the complete pipeline (default)
    python main.py --schedule        # Schedule weekly runs
"""

import os
import logging
import argparse
from datetime import datetime
import scraper
import preprocessor
import model
import scheduler

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'main_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

def find_latest_file(directory, prefix):
    """Find the most recent file with the given prefix in the directory"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    
    # Sort by creation time (newest first)
    return os.path.join(directory, sorted(files, reverse=True)[0])

def run_scraper():
    """Run the scraping process"""
    logger.info("Starting RateMyProfessors scraping process")
    rmp_scraper = scraper.RateMyProfessorsScraper()
    csv_path = rmp_scraper.scrape_penn_state_reviews(max_professors=50, max_reviews_per_professor=20)
    
    if csv_path:
        logger.info(f"Scraping completed successfully. Data saved to {csv_path}")
        return csv_path
    else:
        logger.error("Scraping failed or no reviews were collected.")
        return None

def run_preprocessor(csv_path=None):
    """Run the preprocessing on the most recent scraped data"""
    if not csv_path:
        # Find the most recent scraped data
        csv_path = find_latest_file('data', 'penn_state_reviews_')
        
    if not csv_path:
        logger.error("No scraped data found to process.")
        return None, None
        
    logger.info(f"Starting preprocessing of {csv_path}")
    text_preprocessor = preprocessor.TextPreprocessor()
    processed_path = text_preprocessor.process_reviews_data(csv_path)
    
    if not processed_path:
        logger.error("Preprocessing failed.")
        return None, None
        
    # Prepare for training
    train_path, test_path = text_preprocessor.prepare_for_training(processed_path)
    
    if train_path and test_path:
        logger.info(f"Preprocessing and preparation completed successfully.")
        logger.info(f"Training data: {train_path}")
        logger.info(f"Testing data: {test_path}")
        return train_path, test_path
    else:
        logger.error("Failed to prepare data for training.")
        return None, None

def run_model_training(train_path=None, test_path=None):
    """Train the sentiment model on the most recent processed data"""
    if not train_path:
        # Find the most recent train data
        train_path = os.path.join('data', 'train_data.csv')
        
    if not test_path:
        # Find the most recent test data
        test_path = os.path.join('data', 'test_data.csv')
        
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Training or testing data not found.")
        return None
        
    logger.info(f"Starting model training with {train_path} and {test_path}")
    model_dir = model.fine_tune_model(
        train_path=train_path,
        test_path=test_path,
        epochs=4,
        batch_size=16,
        learning_rate=2e-5
    )
    
    if model_dir:
        logger.info(f"Model training completed successfully. Model saved to {model_dir}")
        return model_dir
    else:
        logger.error("Model training failed.")
        return None

def run_full_pipeline():
    """Run the complete pipeline: scrape -> process -> train"""
    logger.info("Starting the complete sentiment analysis pipeline")
    
    # Step 1: Scrape
    csv_path = run_scraper()
    if not csv_path:
        return
    
    # Step 2: Process
    train_path, test_path = run_preprocessor(csv_path)
    if not train_path or not test_path:
        return
    
    # Step 3: Train
    model_dir = run_model_training(train_path, test_path)
    if not model_dir:
        return
    
    logger.info("Complete pipeline executed successfully")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RateMyProfessors Sentiment Analysis Pipeline')
    
    parser.add_argument('--scrape', action='store_true', help='Run just the scraping process')
    parser.add_argument('--process', action='store_true', help='Process the most recent scraped data')
    parser.add_argument('--train', action='store_true', help='Train on the most recent processed data')
    parser.add_argument('--run-all', action='store_true', help='Run the complete pipeline (default)')
    parser.add_argument('--schedule', action='store_true', help='Schedule weekly runs')
    
    args = parser.parse_args()
    
    # If no arguments provided, default to run-all
    if not any(vars(args).values()):
        args.run_all = True
    
    # Execute based on arguments
    if args.scrape:
        run_scraper()
    
    if args.process:
        run_preprocessor()
    
    if args.train:
        run_model_training()
    
    if args.run_all:
        run_full_pipeline()
    
    if args.schedule:
        logger.info("Starting scheduler for weekly pipeline runs")
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()