import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'performance_tracker_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('performance_tracker')

class ModelPerformanceTracker:
    def __init__(self):
        """Initialize the model performance tracker"""
        self.performance_dir = 'performance_tracking'
        os.makedirs(self.performance_dir, exist_ok=True)
        self.standard_metrics = []  # List to store standard model metrics
        self.combined_metrics = []  # List to store combined model metrics
        logger.info(f"Performance tracker initialized. Plots will be saved to {self.performance_dir}")
    
    def extract_date_from_filename(self, filename):
        """Extract date from a filename with format *_YYYYMMDD.txt"""
        match = re.search(r'_(\d{8})\.txt$', filename)
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                logger.warning(f"Invalid date format in filename: {filename}")
        return None
    
    def parse_metrics_file(self, file_path):
        """Parse a metrics file and extract the metrics"""
        metrics = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Parse each metric line with format "Metric Name: Value"
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        try:
                            value = float(value.strip())
                            metrics[key] = value
                        except ValueError:
                            logger.warning(f"Non-numeric value found for {key}: {value}")
            
            # Add date to metrics
            date = self.extract_date_from_filename(os.path.basename(file_path))
            if date:
                metrics['Date'] = date
            
            return metrics
        except Exception as e:
            logger.error(f"Error parsing metrics file {file_path}: {e}")
            return {}
    
    def collect_metrics(self):
        """Collect metrics from all metrics files in the logs directory"""
        logs_dir = 'logs'
        
        # Collect standard model metrics
        standard_files = [f for f in os.listdir(logs_dir) if f.startswith('standard_test_metrics_') and f.endswith('.txt')]
        for file in standard_files:
            metrics = self.parse_metrics_file(os.path.join(logs_dir, file))
            if metrics:
                metrics['Model Type'] = 'Standard'
                self.standard_metrics.append(metrics)
        
        # Collect combined model metrics
        combined_files = [f for f in os.listdir(logs_dir) if f.startswith('combined_test_metrics_') and f.endswith('.txt')]
        for file in combined_files:
            metrics = self.parse_metrics_file(os.path.join(logs_dir, file))
            if metrics:
                metrics['Model Type'] = 'Combined'
                self.combined_metrics.append(metrics)
        
        # Sort metrics by date
        self.standard_metrics.sort(key=lambda x: x.get('Date', datetime.min))
        self.combined_metrics.sort(key=lambda x: x.get('Date', datetime.min))
        
        # Add run number to metrics
        for i, metrics in enumerate(self.standard_metrics):
            metrics['Run'] = i + 1
            metrics['Training Data'] = f"Run {i + 1}"
        
        for i, metrics in enumerate(self.combined_metrics):
            metrics['Run'] = i + 1
            metrics['Training Data'] = f"Run {i + 1}"
        
        logger.info(f"Collected metrics for {len(self.standard_metrics)} standard model runs and {len(self.combined_metrics)} combined model runs")
    
    def create_metrics_dataframe(self):
        """Create a DataFrame from the collected metrics"""
        # Combine standard and combined metrics
        all_metrics = self.standard_metrics + self.combined_metrics
        
        if not all_metrics:
            logger.warning("No metrics found to create DataFrame")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Ensure required columns exist
        required_columns = ['Date', 'Model Type', 'Run', 'Training Data', 'Test Accuracy', 'Test F1 Score', 'Test Loss']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column {col} not found in metrics data")
                if col not in ['Date', 'Model Type', 'Run', 'Training Data']:
                    df[col] = None
        
        # Create a string version of date for display
        df['Date_Str'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else str(x))
        
        return df
    
    def plot_accuracy_over_time(self, df):
        """Plot model accuracy over time"""
        if df is None or df.empty:
            logger.warning("No data available to plot accuracy over time")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            # Convert dates to matplotlib dates for plotting
            plt.plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test Accuracy'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)  # Accuracy is between 0 and 1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format date labels for better readability
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'accuracy_over_time.png')
        plt.savefig(plot_path)
        logger.info(f"Saved accuracy over time plot to {plot_path}")
    
    def plot_f1_over_time(self, df):
        """Plot model F1 score over time"""
        if df is None or df.empty:
            logger.warning("No data available to plot F1 score over time")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot F1 score for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            plt.plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test F1 Score'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model F1 Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)  # F1 Score is between 0 and 1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format date labels for better readability
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'f1_over_time.png')
        plt.savefig(plot_path)
        logger.info(f"Saved F1 score over time plot to {plot_path}")
    
    def plot_loss_over_time(self, df):
        """Plot model loss over time"""
        if df is None or df.empty:
            logger.warning("No data available to plot loss over time")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot loss for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            plt.plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test Loss'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model Loss Over Time')
        plt.xlabel('Date')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format date labels for better readability
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'loss_over_time.png')
        plt.savefig(plot_path)
        logger.info(f"Saved loss over time plot to {plot_path}")
    
    def plot_combined_metrics(self, df):
        """Plot all metrics on a single plot for easy comparison"""
        if df is None or df.empty:
            logger.warning("No data available to plot combined metrics")
            return
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot accuracy
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[0].plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test Accuracy'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[0].set_title('Model Accuracy Over Time')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_ylim(0, 1)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # Plot F1 score
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[1].plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test F1 Score'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[1].set_title('Model F1 Score Over Time')
        axs[1].set_ylabel('F1 Score')
        axs[1].set_ylim(0, 1)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        # Plot loss
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[2].plot(model_df['Date'].astype('datetime64[ns]'), model_df['Test Loss'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[2].set_title('Model Loss Over Time')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Loss')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend()
        
        # Format date labels for better readability
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'combined_metrics_over_time.png')
        plt.savefig(plot_path)
        logger.info(f"Saved combined metrics plot to {plot_path}")
    
    def plot_accuracy_by_run(self, df):
        """Plot model accuracy by run number"""
        if df is None or df.empty:
            logger.warning("No data available to plot accuracy by run")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            plt.plot(model_df['Run'], model_df['Test Accuracy'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model Accuracy by Training Run')
        plt.xlabel('Training Run')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)  # Accuracy is between 0 and 1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(df['Run'].unique())
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'accuracy_by_run.png')
        plt.savefig(plot_path)
        logger.info(f"Saved accuracy by run plot to {plot_path}")
    
    def plot_f1_by_run(self, df):
        """Plot model F1 score by run number"""
        if df is None or df.empty:
            logger.warning("No data available to plot F1 score by run")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot F1 score for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            plt.plot(model_df['Run'], model_df['Test F1 Score'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model F1 Score by Training Run')
        plt.xlabel('Training Run')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)  # F1 Score is between 0 and 1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(df['Run'].unique())
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'f1_by_run.png')
        plt.savefig(plot_path)
        logger.info(f"Saved F1 score by run plot to {plot_path}")
    
    def plot_loss_by_run(self, df):
        """Plot model loss by run number"""
        if df is None or df.empty:
            logger.warning("No data available to plot loss by run")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot loss for each model type
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            plt.plot(model_df['Run'], model_df['Test Loss'], 
                   marker='o', label=f"{model_type} Model")
        
        plt.title('Model Loss by Training Run')
        plt.xlabel('Training Run')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(df['Run'].unique())
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'loss_by_run.png')
        plt.savefig(plot_path)
        logger.info(f"Saved loss by run plot to {plot_path}")
    
    def plot_combined_metrics_by_run(self, df):
        """Plot all metrics on a single plot by run number for easy comparison"""
        if df is None or df.empty:
            logger.warning("No data available to plot combined metrics by run")
            return
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot accuracy
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[0].plot(model_df['Run'], model_df['Test Accuracy'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[0].set_title('Model Accuracy by Training Run')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_ylim(0, 1)
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # Plot F1 score
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[1].plot(model_df['Run'], model_df['Test F1 Score'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[1].set_title('Model F1 Score by Training Run')
        axs[1].set_ylabel('F1 Score')
        axs[1].set_ylim(0, 1)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        # Plot loss
        for model_type in df['Model Type'].unique():
            model_df = df[df['Model Type'] == model_type]
            axs[2].plot(model_df['Run'], model_df['Test Loss'], 
                      marker='o', label=f"{model_type} Model")
        
        axs[2].set_title('Model Loss by Training Run')
        axs[2].set_xlabel('Training Run')
        axs[2].set_ylabel('Loss')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend()
        
        # Format x-axis to show integer run numbers
        for ax in axs:
            ax.set_xticks(df['Run'].unique())
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.performance_dir, 'combined_metrics_by_run.png')
        plt.savefig(plot_path)
        logger.info(f"Saved combined metrics by run plot to {plot_path}")
    
    def generate_performance_report(self, df):
        """Generate a performance report with latest metrics and trends"""
        if df is None or df.empty:
            logger.warning("No data available to generate performance report")
            return
        
        # Create a markdown report
        report_path = os.path.join(self.performance_dir, 'model_performance_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Model Performance Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Latest Performance Metrics\n\n")
            
            # Get latest metrics for each model type
            for model_type in df['Model Type'].unique():
                model_df = df[df['Model Type'] == model_type].sort_values('Date')
                
                if not model_df.empty:
                    latest = model_df.iloc[-1]
                    f.write(f"### {model_type} Model (Latest: {latest['Date_Str']})\n\n")
                    f.write(f"- Accuracy: {latest['Test Accuracy']:.4f}\n")
                    f.write(f"- F1 Score: {latest['Test F1 Score']:.4f}\n")
                    f.write(f"- Loss: {latest['Test Loss']:.4f}\n\n")
                    
                    # Calculate improvement from previous run, if available
                    if len(model_df) > 1:
                        previous = model_df.iloc[-2]
                        f.write(f"#### Changes from Previous Run ({previous['Date_Str']})\n\n")
                        
                        accuracy_change = latest['Test Accuracy'] - previous['Test Accuracy']
                        f1_change = latest['Test F1 Score'] - previous['Test F1 Score']
                        loss_change = latest['Test Loss'] - previous['Test Loss']
                        
                        f.write(f"- Accuracy: {accuracy_change:.4f} ({'+' if accuracy_change >= 0 else ''}{accuracy_change/previous['Test Accuracy']*100:.2f}%)\n")
                        f.write(f"- F1 Score: {f1_change:.4f} ({'+' if f1_change >= 0 else ''}{f1_change/previous['Test F1 Score']*100:.2f}%)\n")
                        f.write(f"- Loss: {loss_change:.4f} ({'+' if loss_change >= 0 else ''}{loss_change/previous['Test Loss']*100:.2f}%)\n\n")
            
            # Add trend observations
            f.write("## Performance Trends\n\n")
            
            # Compare standard vs combined model if both exist
            if 'Standard' in df['Model Type'].values and 'Combined' in df['Model Type'].values:
                standard_latest = df[df['Model Type'] == 'Standard'].sort_values('Date').iloc[-1]
                combined_latest = df[df['Model Type'] == 'Combined'].sort_values('Date').iloc[-1]
                
                f.write("### Standard vs Combined Model\n\n")
                
                # Compare latest metrics
                accuracy_diff = combined_latest['Test Accuracy'] - standard_latest['Test Accuracy']
                f1_diff = combined_latest['Test F1 Score'] - standard_latest['Test F1 Score']
                
                if accuracy_diff > 0:
                    f.write(f"- The Combined model outperforms the Standard model in accuracy by {accuracy_diff:.4f} ({accuracy_diff/standard_latest['Test Accuracy']*100:.2f}%)\n")
                else:
                    f.write(f"- The Standard model outperforms the Combined model in accuracy by {abs(accuracy_diff):.4f} ({abs(accuracy_diff)/combined_latest['Test Accuracy']*100:.2f}%)\n")
                
                if f1_diff > 0:
                    f.write(f"- The Combined model outperforms the Standard model in F1 score by {f1_diff:.4f} ({f1_diff/standard_latest['Test F1 Score']*100:.2f}%)\n\n")
                else:
                    f.write(f"- The Standard model outperforms the Combined model in F1 score by {abs(f1_diff):.4f} ({abs(f1_diff)/combined_latest['Test F1 Score']*100:.2f}%)\n\n")
            
            # Include recommendations based on the trends
            f.write("## Recommendations\n\n")
            
            if df['Test Accuracy'].max() < 0.7:
                f.write("- The model accuracy is below 70%. Consider additional training data or hyperparameter tuning.\n")
            
            # Save plot file paths for reference
            f.write("\n## Performance Plots\n\n")
            f.write("The following plots show the performance trends over time:\n\n")
            f.write("1. [Accuracy Over Time](accuracy_over_time.png)\n")
            f.write("2. [F1 Score Over Time](f1_over_time.png)\n")
            f.write("3. [Loss Over Time](loss_over_time.png)\n")
            f.write("4. [Combined Metrics](combined_metrics_over_time.png)\n")
        
        logger.info(f"Generated performance report at {report_path}")
        return report_path
    
    def save_metrics_to_csv(self, df):
        """Save the metrics to a CSV file for future reference"""
        if df is None or df.empty:
            logger.warning("No data available to save to CSV")
            return
        
        # Create a copy of the DataFrame with dates as strings for CSV output
        csv_df = df.copy()
        csv_df['Date'] = csv_df['Date_Str']
        csv_df = csv_df.drop('Date_Str', axis=1)
        
        csv_path = os.path.join(self.performance_dir, 'model_performance_history.csv')
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved metrics history to {csv_path}")
    
    def track_performance(self):
        """Track model performance over time and generate visualizations"""
        # Collect metrics from log files
        self.collect_metrics()
        
        # Create metrics DataFrame
        metrics_df = self.create_metrics_dataframe()
        
        if metrics_df is not None and not metrics_df.empty:
            # Generate individual plots
            self.plot_accuracy_over_time(metrics_df)
            self.plot_f1_over_time(metrics_df)
            self.plot_loss_over_time(metrics_df)
            
            # Generate combined plot
            self.plot_combined_metrics(metrics_df)
            
            # Generate plots by run number
            self.plot_accuracy_by_run(metrics_df)
            self.plot_f1_by_run(metrics_df)
            self.plot_loss_by_run(metrics_df)
            self.plot_combined_metrics_by_run(metrics_df)
            
            # Generate performance report
            report_path = self.generate_performance_report(metrics_df)
            
            # Save metrics to CSV
            self.save_metrics_to_csv(metrics_df)
            
            print(f"Performance tracking completed! Report saved to {report_path}")
            return report_path
        else:
            logger.warning("No metrics data available for performance tracking")
            return None

def main():
    tracker = ModelPerformanceTracker()
    tracker.track_performance()

if __name__ == "__main__":
    main()