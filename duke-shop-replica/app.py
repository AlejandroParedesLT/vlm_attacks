"""
Duke Shop Replica - Flask Server
Simple Flask app to serve the Duke Shop HTML page
"""

from flask import Flask, send_from_directory, jsonify, request
import os

app = Flask(__name__)

# Route to serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# API endpoint for visible products metadata (called by JavaScript)
@app.route('/api/visible-products', methods=['POST'])
def visible_products():
    """Receive metadata about which products are currently visible"""
    data = request.json
    visible_ids = data.get('visible', [])
    print(f"📊 Visible products: {visible_ids}")
    return jsonify({"status": "ok", "received": visible_ids})

# Health check endpoint
@app.route('/health')
def health():
    return {'status': 'ok', 'message': 'Duke Shop replica server is running'}

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Duke Shop replica server starting...")
    print("📁 Serving files from the 'static' directory")
    print("🌐 Open http://localhost:5050 in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5050, debug=True)