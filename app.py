from flask import Flask, request, jsonify, render_template
from recommender import get_recommendations
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        preferences = data.get('preferences', '')
        if not preferences:
            return jsonify({'error': 'No preferences provided'}), 400
            
        logger.info("Getting recommendations...")
        recommendations = get_recommendations(preferences)
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        logger.error(f"Error in recommend: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000))) 