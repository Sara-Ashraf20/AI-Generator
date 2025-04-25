from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})  # CORS security

# Initialize Groq client
try:
    llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        timeout=int(os.getenv("TIMEOUT", 30))
    )
    logging.info("✅ Groq client initialized successfully")
except Exception as e:
    logging.error(f"❌ Error initializing Groq: {e}")
    llm = None

@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/api/generate', methods=['POST'])
def generate_description():
    """Generate course description using Groq AI."""
    if not llm:
        return jsonify({"error": "AI service unavailable"}), 503

    try:
        data = request.get_json()
        logging.info(f"📥 Received request: {data}")

        # Validate input
        title = data.get("title", "").strip()
        category = data.get("category", "").strip()
        language = data.get("language", "English").strip()

        if not title or not category:
            return jsonify({"error": "Title and category are required"}), 400

        # Construct prompt
        prompt = f"""
        Generate a professional course description with:
        - Title: {title}
        - Category: {category}
        - Language: {language}

        Include:
        1️⃣ 3-4 detailed paragraphs about the course
        2️⃣ Target audience
        3️⃣ Learning objectives
        4️⃣ A professional and engaging tone
        """

        # Invoke AI model
        response = llm.invoke(prompt)

        # Ensure response is valid
        if not response or not response.content:
            raise ValueError("Empty response from AI model")

        logging.info("✅ Successfully generated course description")
        return jsonify({"description": response.content, "status": "success"})

    except ValueError as ve:
        logging.error(f"⚠️ Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"❌ Internal server error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=os.getenv("DEBUG", "False") == "True")
