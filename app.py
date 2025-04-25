from flask import Flask, render_template, request, jsonify
from recommender import get_recommendations
import logging
import traceback

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
        # Get and validate request data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        preferences = data.get('preferences', '')
        if not preferences:
            return jsonify({'error': 'Preferences cannot be empty'}), 400
            
        # Get recommendations
        recommendations = get_recommendations(preferences)
        
        # Format response
        response_data = {
            'recommendations': recommendations
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        # Log the full error traceback
        logger.error(f"Error in recommend endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a properly formatted error response
        return jsonify({
            'error': f"Error generating recommendations: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 