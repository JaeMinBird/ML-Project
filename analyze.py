import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'analyze_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('analyzer')

class ReviewAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.data = None
        self.plots_dir = 'analysis_plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        logger.info(f"Analyzer initialized. Plots will be saved to {self.plots_dir}")
    
    def load_data(self, csv_path=None):
        """Load the processed data for analysis"""
        if csv_path is None:
            # Find the most recent processed file
            data_dir = 'data'
            processed_files = [f for f in os.listdir(data_dir) 
                              if f.endswith('_processed.csv')]
            
            if not processed_files:
                logger.error("No processed files found to analyze")
                return False
            
            # Sort by creation time (newest first)
            latest_file = sorted(processed_files, reverse=True)[0]
            csv_path = os.path.join(data_dir, latest_file)
        
        try:
            self.data = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.data)} records from {csv_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_difficulty_sentiment_correlation(self):
        """Analyze correlation between difficulty rating and sentiment"""
        if self.data is None or 'difficulty' not in self.data.columns:
            logger.error("Data not loaded or missing difficulty column")
            return
        
        logger.info("Analyzing correlation between difficulty rating and sentiment")
        
        # Plot difficulty vs sentiment
        plt.figure(figsize=(10, 6))
        
        # Group by difficulty and calculate average rating
        difficulty_rating = self.data.groupby('difficulty_level')['rating'].mean().reset_index()
        
        # Sort by difficulty (easy, moderate, hard) - Modified order to match logical progression
        difficulty_order = ['easy', 'moderate', 'hard']
        difficulty_rating['difficulty_level'] = pd.Categorical(
            difficulty_rating['difficulty_level'], 
            categories=difficulty_order, 
            ordered=True
        )
        difficulty_rating = difficulty_rating.sort_values('difficulty_level')
        
        # Create bar chart
        sns.barplot(x='difficulty_level', y='rating', data=difficulty_rating)
        plt.title('Average Rating by Difficulty Level')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Average Rating')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'difficulty_rating_correlation.png')
        plt.savefig(plot_path)
        logger.info(f"Saved difficulty vs rating plot to {plot_path}")
        
        # Show correlation coefficient
        correlation = self.data['difficulty'].corr(self.data['rating'])
        logger.info(f"Correlation coefficient between difficulty and rating: {correlation:.4f}")
        
        # Create a heatmap showing sentiment distribution across difficulty levels
        plt.figure(figsize=(10, 6))
        
        # Ensure the difficulty levels are ordered correctly in the heatmap
        if 'difficulty_level' in self.data.columns:
            self.data['difficulty_level'] = pd.Categorical(
                self.data['difficulty_level'],
                categories=difficulty_order,
                ordered=True
            )
            
        difficulty_sentiment = pd.crosstab(
            self.data['difficulty_level'], 
            self.data['sentiment'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        sns.heatmap(difficulty_sentiment, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Sentiment Distribution by Difficulty Level (%)')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'difficulty_sentiment_heatmap.png')
        plt.savefig(plot_path)
        logger.info(f"Saved difficulty vs sentiment heatmap to {plot_path}")
    
    def analyze_would_take_again(self):
        """Analyze the impact of 'would take again' on ratings"""
        if self.data is None or 'would_take_again_bool' not in self.data.columns:
            logger.error("Data not loaded or missing would_take_again_bool column")
            return
        
        logger.info("Analyzing impact of 'would take again' on ratings")
        
        # Convert would_take_again_bool to string for plotting
        self.data['would_take_again_str'] = self.data['would_take_again_bool'].apply(
            lambda x: 'Yes' if x == True else ('No' if x == False else 'Unknown')
        )
        
        # Group by would_take_again and calculate average rating
        take_again_rating = self.data.groupby('would_take_again_str')['rating'].mean().reset_index()
        
        # Sort by would_take_again (Yes, No, Unknown)
        take_again_order = ['Yes', 'No', 'Unknown']
        take_again_rating['would_take_again_str'] = pd.Categorical(
            take_again_rating['would_take_again_str'], 
            categories=take_again_order, 
            ordered=True
        )
        take_again_rating = take_again_rating.sort_values('would_take_again_str')
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x='would_take_again_str', y='rating', data=take_again_rating)
        plt.title('Average Rating by Would Take Again')
        plt.xlabel('Would Take Again')
        plt.ylabel('Average Rating')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'would_take_again_rating.png')
        plt.savefig(plot_path)
        logger.info(f"Saved would take again vs rating plot to {plot_path}")
        
        # Create a heatmap showing sentiment distribution across would_take_again
        plt.figure(figsize=(10, 6))
        take_again_sentiment = pd.crosstab(
            self.data['would_take_again_str'], 
            self.data['sentiment'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        sns.heatmap(take_again_sentiment, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Sentiment Distribution by Would Take Again (%)')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'would_take_again_sentiment_heatmap.png')
        plt.savefig(plot_path)
        logger.info(f"Saved would take again vs sentiment heatmap to {plot_path}")
    
    def analyze_department_distribution(self):
        """Analyze sentiment distribution across departments"""
        if self.data is None or 'department' not in self.data.columns:
            logger.error("Data not loaded or missing department column")
            return
        
        logger.info("Analyzing sentiment distribution across departments")
        
        # Get departments with at least 5 reviews
        dept_counts = self.data['department'].value_counts()
        significant_depts = dept_counts[dept_counts >= 5].index.tolist()
        
        if not significant_depts:
            logger.warning("No departments have 5 or more reviews. Skipping analysis.")
            return
        
        # Filter data to include only departments with sufficient reviews
        dept_data = self.data[self.data['department'].isin(significant_depts)]
        
        # Calculate average rating by department
        dept_rating = dept_data.groupby('department')['rating'].mean().sort_values(ascending=False)
        
        # Plot average rating by department
        plt.figure(figsize=(12, 6))
        dept_rating.plot(kind='bar')
        plt.title('Average Rating by Department')
        plt.xlabel('Department')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'department_rating.png')
        plt.savefig(plot_path)
        logger.info(f"Saved department vs rating plot to {plot_path}")
        
        # Create a heatmap showing sentiment distribution across departments
        plt.figure(figsize=(14, 10))
        dept_sentiment = pd.crosstab(
            dept_data['department'], 
            dept_data['sentiment'],
            normalize='index'
        ) * 100  # Convert to percentage
        
        sns.heatmap(dept_sentiment, annot=True, fmt='.1f', cmap='viridis')
        plt.title('Sentiment Distribution by Department (%)')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'department_sentiment_heatmap.png')
        plt.savefig(plot_path)
        logger.info(f"Saved department vs sentiment heatmap to {plot_path}")
    
    def analyze_department_sentiment_vector(self):
        """Generate a detailed sentiment vector analysis by department"""
        if self.data is None or 'department' not in self.data.columns:
            logger.error("Data not loaded or missing department column")
            return
        
        logger.info("Generating sentiment vectors by department")
        
        # Get departments with at least 5 reviews
        dept_counts = self.data['department'].value_counts()
        significant_depts = dept_counts[dept_counts >= 5].index.tolist()
        
        if not significant_depts:
            logger.warning("No departments have 5 or more reviews. Skipping analysis.")
            return
        
        # Filter data to include only departments with sufficient reviews
        dept_data = self.data[self.data['department'].isin(significant_depts)]
        
        # Create vectors of sentiment metrics by department
        dept_metrics = []
        for dept in significant_depts:
            dept_subset = dept_data[dept_data['department'] == dept]
            
            # Calculate metrics
            metrics = {
                'department': dept,
                'review_count': len(dept_subset),
                'avg_rating': dept_subset['rating'].mean(),
                'avg_difficulty': dept_subset['difficulty'].mean(),
                'positive_sentiment': (dept_subset['sentiment'] == 'positive').mean() * 100,
                'neutral_sentiment': (dept_subset['sentiment'] == 'neutral').mean() * 100,
                'negative_sentiment': (dept_subset['sentiment'] == 'negative').mean() * 100,
                'would_take_again': dept_subset['would_take_again_bool'].mean() * 100 if 'would_take_again_bool' in dept_subset.columns else None
            }
            dept_metrics.append(metrics)
        
        # Create DataFrame from metrics
        dept_df = pd.DataFrame(dept_metrics)
        
        # Sort by average rating
        dept_df = dept_df.sort_values('avg_rating', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.plots_dir, 'department_sentiment_vector.csv')
        dept_df.to_csv(csv_path, index=False)
        logger.info(f"Saved department sentiment vector data to {csv_path}")
        
        # Create heatmap of department sentiment vectors
        plt.figure(figsize=(14, 10))
        
        # Select columns for heatmap (excluding department name and review count)
        heatmap_columns = ['avg_rating', 'avg_difficulty', 'positive_sentiment', 
                           'neutral_sentiment', 'negative_sentiment']
        if 'would_take_again' in dept_df.columns and dept_df['would_take_again'].notna().all():
            heatmap_columns.append('would_take_again')
        
        # Normalize the data for better visualization
        heatmap_data = dept_df[heatmap_columns].copy()
        for col in heatmap_data.columns:
            heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())
        
        # Create heatmap with department names as y-axis
        heatmap_data_indexed = heatmap_data.copy()
        heatmap_data_indexed.index = dept_df['department']
        
        sns.heatmap(heatmap_data_indexed, annot=True, fmt='.2f', cmap='viridis')
        plt.title('Department Sentiment Vector Analysis (Normalized)')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'department_sentiment_vector.png')
        plt.savefig(plot_path)
        logger.info(f"Saved department sentiment vector plot to {plot_path}")
        
        return dept_df
    
    def analyze_tag_distribution(self):
        """Analyze the distribution of tags in reviews"""
        if self.data is None or 'tags' not in self.data.columns:
            logger.error("Data not loaded or missing tags column")
            return
        
        logger.info("Analyzing tag distribution in reviews")
        
        # If we have the tag feature columns
        tag_columns = [col for col in self.data.columns if col.startswith('is_') or 
                       col in ['tough_grader', 'get_ready_to_read', 'skip_class']]
        
        if tag_columns:
            # Calculate the frequency of each tag
            tag_counts = self.data[tag_columns].sum().sort_values(ascending=False)
            
            # Plot only the top 5 tags
            plt.figure(figsize=(12, 6))
            tag_counts.head(5).plot(kind='bar')
            plt.title('Top 5 Most Common Tags in Reviews')
            plt.xlabel('Tag')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'tag_frequency.png')
            plt.savefig(plot_path)
            logger.info(f"Saved top 5 tags frequency plot to {plot_path}")
            
            # Calculate average rating for reviews with each tag
            tag_ratings = {}
            for tag in tag_columns:
                avg_rating = self.data[self.data[tag] == True]['rating'].mean()
                tag_ratings[tag] = avg_rating
            
            # Plot average rating by tag (top 5 most common tags)
            plt.figure(figsize=(12, 6))
            top5_tags = tag_counts.head(5).index.tolist()
            tag_ratings_series = pd.Series({tag: tag_ratings[tag] for tag in top5_tags}).sort_values(ascending=False)
            tag_ratings_series.plot(kind='bar')
            plt.title('Average Rating for Top 5 Most Common Tags')
            plt.xlabel('Tag')
            plt.ylabel('Average Rating')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'tag_rating.png')
            plt.savefig(plot_path)
            logger.info(f"Saved top 5 tags vs rating plot to {plot_path}")
        else:
            # Parse tags from the comma-separated string
            all_tags = []
            for tags_str in self.data['tags'].dropna():
                if isinstance(tags_str, str):
                    tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    all_tags.extend(tags)
            
            # Count tag frequency
            tag_freq = pd.Series(all_tags).value_counts()
            
            # Plot only the top 5 tags
            plt.figure(figsize=(12, 6))
            tag_freq.head(5).plot(kind='bar')
            plt.title('Top 5 Most Common Tags in Reviews')
            plt.xlabel('Tag')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'top_tags.png')
            plt.savefig(plot_path)
            logger.info(f"Saved top 5 tags plot to {plot_path}")
    
    def analyze_sentiment_distribution(self):
        """Analyze the distribution of sentiment in reviews"""
        if self.data is None or 'sentiment' not in self.data.columns:
            logger.error("Data not loaded or missing sentiment column")
            return
        
        logger.info("Analyzing sentiment distribution in reviews")
        
        # Calculate sentiment distribution
        sentiment_counts = self.data['sentiment'].value_counts()
        
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts.plot(kind='bar')
        plt.title('Sentiment Distribution in Reviews')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'sentiment_distribution.png')
        plt.savefig(plot_path)
        logger.info(f"Saved sentiment distribution plot to {plot_path}")
        
        # If we have combined sentiment, compare with standard sentiment
        if 'combined_sentiment' in self.data.columns:
            # Calculate combined sentiment distribution
            combined_sentiment_counts = self.data['combined_sentiment'].value_counts()
            
            # Plot comparison
            plt.figure(figsize=(12, 6))
            
            # Create a DataFrame for grouped bar chart
            comparison_df = pd.DataFrame({
                'Standard': sentiment_counts,
                'Combined': combined_sentiment_counts
            })
            
            comparison_df.plot(kind='bar')
            plt.title('Comparison of Standard vs Combined Sentiment')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.plots_dir, 'sentiment_comparison.png')
            plt.savefig(plot_path)
            logger.info(f"Saved sentiment comparison plot to {plot_path}")
    
    def run_all_analyses(self):
        """Run all available analyses"""
        logger.info("Running all analyses")
        
        self.analyze_sentiment_distribution()
        self.analyze_difficulty_sentiment_correlation()
        self.analyze_would_take_again()
        self.analyze_department_distribution()
        self.analyze_department_sentiment_vector()
        self.analyze_tag_distribution()
        
        logger.info("All analyses completed. Plots saved to " + self.plots_dir)
        print(f"Analysis complete! All plots saved to {self.plots_dir} directory.")
        
        return self.plots_dir

def main():
    analyzer = ReviewAnalyzer()
    
    if analyzer.load_data():
        analyzer.run_all_analyses()
    else:
        logger.error("Failed to load data. Analysis aborted.")

if __name__ == "__main__":
    main()