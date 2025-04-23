import os
import logging
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import scraper
import preprocessor
import model
import analyze
import compare_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'scheduler_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('scheduler')

def scrape_and_process():
    """
    Scrape RateMyProfessors, process data, and save to CSV
    """
    logger.info("Starting scheduled scraping job")
    
    # Scrape professor reviews
    rmp_scraper = scraper.RateMyProfessorsScraper()
    csv_path = rmp_scraper.scrape_penn_state_reviews(max_professors=50, max_reviews_per_professor=20)
    
    if not csv_path:
        logger.error("Scraping failed or no reviews were collected.")
        return None, None
    
    logger.info(f"Scraping completed. Data saved to {csv_path}")
    
    # Process the scraped data
    text_preprocessor = preprocessor.TextPreprocessor()
    processed_path = text_preprocessor.process_reviews_data(csv_path)
    
    if not processed_path:
        logger.error("Data processing failed.")
        return csv_path, None
    
    logger.info(f"Data processing completed. Processed data saved to {processed_path}")
    
    # Prepare data for training
    train_path, test_path = text_preprocessor.prepare_for_training(processed_path)
    
    if not train_path or not test_path:
        logger.error("Failed to prepare data for training.")
        return csv_path, processed_path
    
    logger.info(f"Data prepared for training. Train data: {train_path}, Test data: {test_path}")
    
    return train_path, test_path

def train_sentiment_model(train_path, test_path):
    """
    Fine-tune sentiment analysis model on the prepared data
    """
    if not train_path or not test_path:
        logger.error("Missing training or test data paths.")
        return None, None
    
    logger.info("Starting scheduled model training job")
    
    # Fine-tune the standard model
    standard_model_dir = model.fine_tune_model(
        train_path=train_path,
        test_path=test_path,
        epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        model_type="standard"
    )
    
    if not standard_model_dir:
        logger.error("Standard model training failed.")
        return None, None
    
    logger.info(f"Standard model training completed. Model saved to {standard_model_dir}")
    
    # Check if we have combined training data
    combined_train_path = os.path.join(os.path.dirname(train_path), 'combined_train_data.csv')
    combined_test_path = os.path.join(os.path.dirname(test_path), 'combined_test_data.csv')
    
    if os.path.exists(combined_train_path) and os.path.exists(combined_test_path):
        # Fine-tune the combined model
        combined_model_dir = model.fine_tune_combined_model(
            train_path=combined_train_path,
            test_path=combined_test_path,
            epochs=4,
            batch_size=16,
            learning_rate=2e-5
        )
        
        if not combined_model_dir:
            logger.error("Combined model training failed.")
            return standard_model_dir, None
        
        logger.info(f"Combined model training completed. Model saved to {combined_model_dir}")
        return standard_model_dir, combined_model_dir
    else:
        logger.info("No combined training data found. Skipping combined model training.")
        return standard_model_dir, None

def analyze_results():
    """
    Run analysis on the processed data and models
    """
    logger.info("Starting analysis job")
    
    # Analyze data
    analyzer = analyze.ReviewAnalyzer()
    if analyzer.load_data():
        analyzer.run_all_analyses()
        logger.info("Data analysis completed")
    else:
        logger.error("Data analysis failed")
    
    # Compare models
    comparator = compare_models.ModelComparator()
    if comparator.load_models() and comparator.load_test_data():
        summary = comparator.run_comparison()
        logger.info("Model comparison completed")
        return summary
    else:
        logger.error("Model comparison failed")
        return None

def track_model_performance():
    """
    Track model performance over time
    """
    logger.info("Tracking model performance")
    
    # Find all comparison summaries
    plots_dir = 'model_comparison_plots'
    if not os.path.exists(plots_dir):
        logger.warning("No model comparison directory found")
        return
    
    summary_files = [f for f in os.listdir(plots_dir) if f.startswith('model_comparison_summary') and f.endswith('.json')]
    if not summary_files:
        logger.warning("No model comparison summaries found")
        return
    
    # Load all summaries
    summaries = []
    for summary_file in summary_files:
        try:
            with open(os.path.join(plots_dir, summary_file), 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
        except Exception as e:
            logger.error(f"Error loading summary file {summary_file}: {e}")
    
    if not summaries:
        logger.warning("No valid summaries found")
        return
    
    # Sort by timestamp
    summaries.sort(key=lambda x: x.get('timestamp', ''))
    
    # Extract performance metrics over time
    timestamps = [summary.get('timestamp', 'Unknown') for summary in summaries]
    standard_accuracies = [summary.get('standard_model_accuracy', 0) for summary in summaries]
    combined_accuracies = [summary.get('combined_model_accuracy', 0) for summary in summaries]
    
    # Plot accuracy over time
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, standard_accuracies, marker='o', label='Standard Model')
    plt.plot(timestamps, combined_accuracies, marker='x', label='Combined Model')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    performance_plot_path = os.path.join('analysis_plots', 'model_performance_over_time.png')
    plt.savefig(performance_plot_path)
    logger.info(f"Saved model performance plot to {performance_plot_path}")

def full_pipeline_job():
    """
    Run the full pipeline: scrape -> process -> train -> analyze
    """
    logger.info("Starting full sentiment analysis pipeline")
    
    # Step 1: Scrape and process data
    train_path, test_path = scrape_and_process()
    
    if not train_path or not test_path:
        logger.error("Scraping and processing step failed. Aborting pipeline.")
        return
    
    # Step 2: Train models
    standard_model_dir, combined_model_dir = train_sentiment_model(train_path, test_path)
    
    if not standard_model_dir:
        logger.error("Model training step failed. Pipeline did not complete successfully.")
        return
    
    # Step 3: Analyze results
    analyze_results()
    
    # Step 4: Track model performance over time
    track_model_performance()
    
    logger.info(f"Full pipeline completed successfully")

def start_scheduler():
    """
    Start the background scheduler to run tasks on a schedule
    """
    scheduler = BackgroundScheduler()
    
    # Schedule the full pipeline to run daily at 1 AM
    scheduler.add_job(
        full_pipeline_job,
        trigger=CronTrigger(hour=1, minute=0),
        id='sentiment_pipeline',
        name='Daily sentiment analysis pipeline',
        max_instances=1,
        replace_existing=True
    )
    
    # Start the scheduler
    try:
        scheduler.start()
        logger.info("Scheduler started. The pipeline will run daily at 1 AM.")
        logger.info("Press Ctrl+C to exit")
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
        scheduler.shutdown()

def run_once():
    """
    Run the full pipeline once immediately
    """
    logger.info("Running the sentiment analysis pipeline once")
    full_pipeline_job()
    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    import argparse
    import json
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='RateMyProfessors Sentiment Analysis Scheduler')
    parser.add_argument('--run-once', action='store_true', help='Run the pipeline once and exit')
    parser.add_argument('--scrape-only', action='store_true', help='Run only the scraping and processing step')
    parser.add_argument('--train-only', action='store_true', help='Run only the model training step')
    parser.add_argument('--analyze-only', action='store_true', help='Run only the analysis step')
    
    args = parser.parse_args()
    
    if args.run_once:
        run_once()
    elif args.scrape_only:
        scrape_and_process()
    elif args.train_only:
        # Find the most recent train/test data
        data_dir = 'data'
        train_path = os.path.join(data_dir, 'train_data.csv')
        test_path = os.path.join(data_dir, 'test_data.csv')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_sentiment_model(train_path, test_path)
        else:
            logger.error("Training data not found")
    elif args.analyze_only:
        analyze_results()
    else:
        start_scheduler()