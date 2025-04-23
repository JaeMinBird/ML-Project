# Penn State Rate My Professors - Sentiment Analysis Classifier

This project scrapes professor reviews from Rate My Professors for Penn State University, processes the data to extract valuable insights, and trains sentiment analysis models to understand student opinions. The project includes data analysis and visualization capabilities, along with model comparison tools to evaluate different sentiment prediction approaches.

## Project Overview

The sentiment analysis pipeline consists of several components that work together:

1. **Web Scraping**: Collects professor reviews from RateMyProfessors.com using the site's embedded data structure
2. **Data Preprocessing**: Cleans and transforms the raw review data, extracting advanced features
3. **Model Training**: Creates two sentiment models:
   - Standard Model: Based on review ratings only
   - Combined Model: Incorporates difficulty ratings and other metadata
4. **Analysis & Visualization**: Generates insights and visualizations about professor reviews
5. **Model Comparison**: Evaluates and compares the performance of both models
6. **Automated Pipeline**: Runs daily to continuously improve the models with new data
7. **Performance Tracking**: Tracks model performance over time with visualizations and reports
8. **Interactive Web Dashboard**: Flask-based UI for exploring data and visualization results

## Features

- **Enhanced Data Collection**: Scrapes not just review text and ratings, but also difficulty scores, tags, and "would take again" information
- **Dynamic Tag Extraction**: Automatically extracts professor's top tags from their profile pages
- **Randomized Professor Selection**: Avoids repetition of professors in scraping cycles
- **Nuanced Sentiment Analysis**: Creates a combined score that balances rating with difficulty
- **Tag Analysis**: Extracts meaningful features from review tags (helpful, tough, caring, etc.)
- **Department Analysis**: Identifies sentiment patterns across different academic departments
- **Model Comparison**: Analyzes where and why the standard and combined models differ in predictions
- **Performance Tracking**: Visual tracking of model metrics over time to measure continuous improvement
- **Automated Learning**: Daily scheduled runs to continuously improve model accuracy

## Project Structure

```
📁 ML Project/
├── 📁 data/                  # Stores scraped and processed data
├── 📁 models/                # Saved trained models
├── 📁 logs/                  # Application logs
├── 📁 analysis_plots/        # Generated data visualizations
├── 📁 model_comparison_plots/# Model comparison analysis
├── 📁 performance_tracking/  # Performance metrics and trends over time
├── 📁 screenshots/           # UI screenshots for documentation
├── 📁 static/                # Static assets for the Flask web application
├── 📁 templates/             # HTML templates for the Flask web application
├── 📄 scraper.py             # Rate My Professors web scraper
├── 📄 preprocessor.py        # Data cleaning and feature extraction
├── 📄 model.py               # Sentiment analysis model training
├── 📄 analyze.py             # Data analysis and visualization
├── 📄 compare_models.py      # Model comparison tools
├── 📄 track_model_performance.py # Performance tracking over time
├── 📄 scheduler.py           # Automated pipeline scheduler
├── 📄 app.py                 # Flask web application
├── 📄 main.py                # Main application entry point
└── 📄 requirements.txt       # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`):
  - requests
  - pandas
  - numpy
  - beautifulsoup4
  - transformers
  - torch
  - scikit-learn
  - matplotlib
  - seaborn
  - apscheduler
  - tqdm
  - regex
  - json
  - logging
  - datetime
  - flask
  - plotly
  - plotly-express

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ML Project
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create required directories:
   ```
   mkdir -p data logs models analysis_plots model_comparison_plots performance_tracking
   ```

### Usage

#### Running the Complete Pipeline

To run the complete pipeline (scrape -> process -> train -> analyze):

```
python main.py --run-all
```

#### Running Individual Components

To run specific parts of the pipeline:

```
python main.py --scrape          # Run just the scraping process
python main.py --process         # Process the most recent scraped data
python main.py --train           # Train on the most recent processed data
```

#### Data Analysis

To analyze the processed data and generate visualizations:

```
python analyze.py
```

This will create visualizations in the `analysis_plots` directory, including:
- Sentiment distribution by difficulty level
- Department rating comparisons
- Tag frequency analysis
- "Would take again" correlations

#### Model Comparison

To compare the standard and combined sentiment models:

```
python compare_models.py
```

This creates comparison reports in the `model_comparison_plots` directory, showing:
- Accuracy comparison between models
- Analysis of cases where models disagree
- Correlation between disagreements and review difficulty

#### Performance Tracking

To track model performance metrics over time:

```
python track_model_performance.py
```

This generates visualizations and reports in the `performance_tracking` directory, showing:
- Model accuracy trends over time
- F1 score progression
- Loss reduction
- Comparative analysis between standard and combined models

#### Scheduled Execution

To set up automated daily runs:

```
python scheduler.py
```

Or to run the scheduler with specific components only:

```
python scheduler.py --scrape-only
python scheduler.py --train-only
python scheduler.py --analyze-only
```

## Web Application Interface

The project includes an interactive Flask-based web application that provides a user-friendly interface for exploring the sentiment analysis data and visualizations. The web application allows users to:

- View dashboard summary metrics
- Explore interactive charts with customizable parameters
- Browse and filter the review data
- Compare departments and professors

To run the web application:

```
python app.py
```

Then open your browser and navigate to `http://localhost:5000`.

### Dashboard Overview

The dashboard provides a high-level summary of the Penn State RMP data analysis, including key metrics such as total number of reviews, professors analyzed, average ratings, and sentiment distribution. It offers a quick snapshot of the current state of the dataset.

![Dashboard Overview](screenshots/dashboard.png)
*The dashboard overview shows key metrics including review count, professor count, average ratings, and sentiment distribution. Users can quickly gauge the overall tone of professor reviews across Penn State University.*

### Interactive Charts

The application features a powerful interactive charting system that allows users to create custom visualizations by selecting different variables, chart types, aggregation methods, and filters.

![Department Rating Chart](screenshots/ChartEx1.png)
*This chart example shows the average rating by department, broken down by sentiment. Users can customize the x-axis, y-axis, color grouping, chart type, and apply various filters to explore different aspects of the data.*

![Rating vs Difficulty Chart](screenshots/ChartEx2.png)
*This plot demonstrates the relationship between professor ratings and difficulty scores. Each point represents a professor, with colors indicating sentiment categories. This visualization helps identify patterns between perceived difficulty and overall professor ratings.*

### Data Browser

The data browser section allows users to explore the underlying review data directly. Users can filter reviews by department and professor, see the full text of reviews, and analyze individual ratings, difficulties, and sentiments.

![Data Browser](screenshots/data_browser.png)
*The data browser provides direct access to individual reviews. Users can filter by department and professor to find specific reviews of interest, showing detailed information including the review text, rating, difficulty score, and sentiment classification.*

The web application makes the entire sentiment analysis pipeline accessible to non-technical users, enabling broader engagement with the insights generated from the Rate My Professors data. It serves as both a research tool and a practical utility for students looking to make informed decisions about course selection.

## Key Insights

### Sentiment vs. Difficulty

![Difficulty Analysis](analysis_plots/difficulty_sentiment_heatmap.png)

The analysis reveals that professor difficulty has a substantial impact on sentiment. The combined model weights difficulty appropriately to provide a more balanced view of professor quality.

### Department Comparison

![Department Analysis](analysis_plots/department_rating.png)

Sentiment varies significantly across departments, with some consistently receiving more positive reviews than others.

### Tag Analysis

![Tag Analysis](analysis_plots/tag_frequency.png)

The most common tags provide insight into what students value most in their professors, with "helpful" being particularly associated with positive reviews.

### Model Performance Over Time

![Model Performance](performance_tracking/combined_metrics_over_time.png)

As more data is collected daily, both models show continuous improvement, with the combined model generally outperforming the standard model. The performance tracking feature helps visualize these trends over time.

## Future Directions

1. **Multi-Institution Analysis**: Expand to compare professor reviews across multiple universities
2. **Temporal Analysis**: Track how sentiment changes over academic terms
3. **NLP Model Improvements**: Experiment with larger language models for increased accuracy
4. **Interactive Dashboard**: Create a web interface for exploring the data and models
5. **Personalized Recommendations**: Develop a system to recommend professors based on student preferences
6. **Advanced Tag Analysis**: Deeper analysis of tag correlations with sentiment and department

## Recent Updates

- **April 2025**: 
  - Enhanced tag extraction from professor profile pages
  - Added randomization to avoid repeating professors in scrapes
  - Fixed difficulty level ordering in charts for better interpretability
  - Added model performance tracking over time
  - Updated README with comprehensive library documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rate My Professors for providing the source data
- Penn State University professors and students who contributed reviews
- The open-source community for the excellent NLP and data science tools