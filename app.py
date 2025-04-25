from flask import Flask, render_template, request, jsonify
from recommender import get_recommendations
import logging
import traceback
import json

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
        logger.info("Received recommendation request")
        
        # Get and validate request data
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        logger.info(f"Request data: {json.dumps(data)}")
        
        if not data:
            logger.error("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
            
        preferences = data.get('preferences', '')
        if not preferences:
            logger.error("Empty preferences provided")
            return jsonify({'error': 'Preferences cannot be empty'}), 400
            
        logger.info("Getting recommendations...")
        # Get recommendations
        recommendations = get_recommendations(preferences)
        logger.info(f"Got {len(recommendations)} recommendations")
        
        # Format response
        response_data = {
            'recommendations': recommendations
        }
        
        logger.info("Sending response")
        return jsonify(response_data)
        
    except Exception as e:
        # Log the full error traceback
        logger.error(f"Error in recommend endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return a properly formatted error response
        error_response = {
            'error': f"Error generating recommendations: {str(e)}"
        }
        logger.error(f"Sending error response: {json.dumps(error_response)}")
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(debug=True) 