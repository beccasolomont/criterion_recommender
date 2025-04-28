from flask import Flask, request, jsonify, render_template
from recommender import get_recommendations
import logging
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize embeddings at startup
try:
    logger.info("Initializing embeddings...")
    get_recommendations("test")  # This will trigger embedding initialization
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Error initializing embeddings: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        preferences = data.get('preferences', '')
        if not preferences:
            return jsonify({'error': 'No preferences provided'}), 400
            
        logger.info("Getting recommendations...")
        recommendations = get_recommendations(preferences)
        
        # Log performance metrics
        duration = time.time() - start_time
        logger.info(f"Recommendation generated in {duration:.2f} seconds")
        
        return jsonify({
            'recommendations': recommendations,
            'duration': f"{duration:.2f}s"
        })
        
    except Exception as e:
        logger.error(f"Error in recommend: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000))) 