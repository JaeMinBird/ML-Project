import os
import pandas as pd
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('flask_app')

# Initialize Flask app
app = Flask(__name__)

def load_latest_data():
    """Load the most recent processed data file"""
    data_dir = 'data'
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    
    if not processed_files:
        logger.warning("No processed data files found")
        return None
    
    # Sort by creation time (newest first)
    latest_file = sorted(processed_files, reverse=True)[0]
    csv_path = os.path.join(data_dir, latest_file)
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    df = load_latest_data()
    current_year = datetime.now().year
    
    if df is None:
        return render_template('index.html', 
                               data_available=False, 
                               error_message="No processed data available",
                               current_year=current_year)
    
    # Get list of departments for filtering
    departments = sorted(df['department'].unique().tolist())
    
    # Calculate summary statistics
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    avg_rating = df['rating'].mean()
    avg_difficulty = df['difficulty'].mean() if 'difficulty' in df.columns else None
    review_count = len(df)
    professor_count = df['professor_name'].nunique()
    
    return render_template('index.html',
                          data_available=True,
                          departments=departments,
                          sentiment_counts=sentiment_counts,
                          avg_rating=avg_rating,
                          avg_difficulty=avg_difficulty,
                          review_count=review_count,
                          professor_count=professor_count,
                          current_year=current_year)

@app.route('/data')
def get_data():
    """API endpoint to get filtered data"""
    df = load_latest_data()
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    # Apply filters
    department = request.args.get('department')
    sentiment = request.args.get('sentiment')
    difficulty = request.args.get('difficulty')
    
    filtered_df = df.copy()
    
    if department and department != 'All':
        filtered_df = filtered_df[filtered_df['department'] == department]
    
    if sentiment and sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
    
    if difficulty and difficulty != 'All' and 'difficulty_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['difficulty_level'] == difficulty]
    
    # Convert to dictionary for JSON serialization
    result = filtered_df.head(100).to_dict(orient='records')
    return jsonify(result)

@app.route('/chart-data')
def get_chart_data():
    """API endpoint to get data for charts"""
    df = load_latest_data()
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    # Apply filters
    department = request.args.get('department')
    sentiment = request.args.get('sentiment')
    difficulty = request.args.get('difficulty')
    
    filtered_df = df.copy()
    
    if department and department != 'All':
        filtered_df = filtered_df[filtered_df['department'] == department]
    
    if sentiment and sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
    
    if difficulty and difficulty != 'All' and 'difficulty_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['difficulty_level'] == difficulty]
    
    return jsonify(filtered_df.to_dict(orient='records'))

@app.route('/create-chart')
def create_chart():
    """API endpoint to create charts with user-selected variables"""
    df = load_latest_data()
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    # Get chart parameters from request
    x_axis = request.args.get('x_axis', 'department')
    y_axis = request.args.get('y_axis', 'rating')
    chart_type = request.args.get('chart_type', 'bar')
    color_by = request.args.get('color_by', 'sentiment')
    
    # Apply filters
    department = request.args.get('department')
    sentiment = request.args.get('sentiment')
    difficulty = request.args.get('difficulty')
    
    filtered_df = df.copy()
    
    if department and department != 'All':
        filtered_df = filtered_df[filtered_df['department'] == department]
    
    if sentiment and sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
    
    if difficulty and difficulty != 'All' and 'difficulty_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['difficulty_level'] == difficulty]
    
    # Handle aggregation for y-axis if needed
    agg_func = request.args.get('agg_func', 'mean')
    
    # Create the appropriate chart based on chart type
    try:
        if chart_type == 'bar':
            # For bar charts, we need to aggregate
            if agg_func == 'count':
                agg_df = filtered_df.groupby(x_axis).size().reset_index(name=y_axis)
                fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f'Count of {x_axis}')
            else:
                agg_df = filtered_df.groupby([x_axis, color_by])[y_axis].agg(agg_func).reset_index()
                fig = px.bar(agg_df, x=x_axis, y=y_axis, color=color_by, 
                             title=f'{agg_func.capitalize()} {y_axis} by {x_axis}')
        
        elif chart_type == 'scatter':
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                            hover_data=['professor_name', 'clean_text'],
                            title=f'{y_axis} vs {x_axis} colored by {color_by}')
        
        elif chart_type == 'box':
            fig = px.box(filtered_df, x=x_axis, y=y_axis, color=color_by,
                        title=f'{y_axis} distribution by {x_axis}')
        
        elif chart_type == 'heatmap':
            # For heatmap, we need a different approach
            heatmap_df = pd.crosstab(filtered_df[x_axis], filtered_df[color_by])
            fig = px.imshow(heatmap_df, title=f'Heatmap of {x_axis} vs {color_by}')
        
        else:
            # Default to bar chart
            agg_df = filtered_df.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
            fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f'{agg_func.capitalize()} {y_axis} by {x_axis}')
        
        # Add layout improvements
        fig.update_layout(
            template='plotly_white',
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title=y_axis.replace('_', ' ').title()
        )
        
        # Convert to JSON for return
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'chart': chart_json})
    
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return jsonify({'error': f'Error creating chart: {str(e)}'})

@app.route('/available-fields')
def get_available_fields():
    """Return available fields for chart axes"""
    df = load_latest_data()
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    # Get numerical columns for y-axis
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Get categorical columns for x-axis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Filter out large text fields and irrelevant columns
    exclude_cols = ['clean_text', 'text', 'enhanced_text', 'train_text', 'tags', 
                    'professor_top_tags', 'tags_list']
    
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    return jsonify({
        'categorical': categorical_cols,
        'numerical': numerical_cols
    })

@app.route('/professors')
def get_professors():
    """Get list of professors by department"""
    df = load_latest_data()
    
    if df is None:
        return jsonify({'error': 'No data available'})
    
    department = request.args.get('department')
    
    if not department or department == 'All':
        professors = sorted(df['professor_name'].unique().tolist())
    else:
        professors = sorted(df[df['department'] == department]['professor_name'].unique().tolist())
    
    return jsonify(professors)

# Add a context processor to make current_year available to all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

if __name__ == '__main__':
    app.run(debug=True)