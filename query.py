import os
import json
import vertexai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
from google.cloud import storage
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Load Configuration from JSON
# ==============================
CONFIG_PATH = "embeddings_config.json"

# Global variables to cache config and embeddings
_config = None
_cached_embeddings = None
_embedding_model = None

def load_config():
    """Load configuration from embeddings config file."""
    global _config
    
    if _config is not None:
        return _config
        
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                _config = json.load(f)
            logger.info("Configuration loaded successfully")
            return _config
        else:
            logger.error(f"Config file {CONFIG_PATH} not found. Please run uploader.py first.")
            return None
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return None

def initialize_services():
    """Initialize Vertex AI and other services. Call this once at startup."""
    global _embedding_model
    
    config = load_config()
    if not config:
        raise Exception("Failed to load configuration")

    PROJECT_ID = config["project_id"]
    LOCATION = config["location"]

    # Initialize Vertex AI
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
        logger.info("Vertex AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        raise

def load_embeddings_from_gcs():
    """
    Load product embeddings from Google Cloud Storage with caching.
    """
    global _cached_embeddings
    
    # Return cached embeddings if available
    if _cached_embeddings is not None:
        return _cached_embeddings
    
    config = load_config()
    if not config:
        return {}
    
    try:
        PROJECT_ID = config["project_id"]
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(config["gcs_bucket"])
        
        # Construct the full path
        gcs_file_path = f"{config['gcs_folder']}/{config['gcs_file']}"
        blob = bucket.blob(gcs_file_path)
        
        # Download the JSON content
        embeddings_data = json.loads(blob.download_as_text())
        
        # Cache the embeddings
        _cached_embeddings = embeddings_data
        
        logger.info(f"Successfully loaded {len(embeddings_data)} sunglasses products from GCS")
        
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Error loading embeddings from GCS: {e}")
        return {}

def query_realtime_embeddings(user_query, neighbor_count=5):
    """
    Query sunglasses products using real-time embedding computation and cosine similarity.
    """
    try:
        if not user_query or not user_query.strip():
            logger.warning("Empty or invalid search query provided")
            return []

        if neighbor_count < 1:
            neighbor_count = 5

        # Ensure services are initialized
        if _embedding_model is None:
            initialize_services()

        # Load stored product embeddings from GCS
        product_embeddings = load_embeddings_from_gcs()
        
        if not product_embeddings:
            logger.warning("No product embeddings available")
            return []

        # Generate embedding for user query
        query_embedding = _embedding_model.get_embeddings([user_query])[0].values
        
        # Calculate similarities
        similarities = []
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        
        for product_id, product_data in product_embeddings.items():
            try:
                product_embedding = np.array(product_data["embedding"]).reshape(1, -1)
                metadata = product_data.get("metadata", {})
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding_array, product_embedding)[0][0]
                
                similarities.append({
                    'product_id': product_id,
                    'similarity_score': float(similarity),
                    'distance': 1 - float(similarity),
                    'metadata': metadata
                })
                
            except Exception as e:
                logger.error(f"Error processing product {product_id}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Found {len(similarities)} products for query: '{user_query}'")
        
        # Return top N results
        return similarities[:neighbor_count]

    except Exception as e:
        logger.error(f"Error querying real-time embeddings: {str(e)}")
        return []

def generate_llm_response(user_query, search_results):
    """
    Generate a natural language response using an LLM with actual product names.
    Returns a dictionary (not JSON string) for easier handling.
    """
    try:
        if not search_results:
            fallback_response = {
                "response": "I'm Yris, your AI shopping assistant! I couldn't find specific matches for that query, but I'd be happy to help you explore our sunglasses collection. We offer styles ranging from classic to trendy, all with premium UV protection. What type of look are you going for?",
                "action": "generalquery"
            }
            return fallback_response

        # Try Vertex AI Gemini models first
        try:
            model_versions = ["gemini-2.5-flash"]
            for version in model_versions:
                try:
                    model = GenerativeModel(version)
                    response_dict = _generate_with_vertex_gemini(model, user_query, search_results)
                    if response_dict:
                        return response_dict
                except Exception as e:
                    logger.warning(f"Failed with {version}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Vertex AI Gemini attempt failed: {e}")

        # Try Gemini API as backup
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                response_dict = _generate_with_gemini_api(model, user_query, search_results)
                if response_dict:
                    return response_dict
        except Exception as e:
            logger.warning(f"Gemini API attempt failed: {e}")

        # Enhanced fallback with real product names
        top_result = search_results[0]
        product_name = top_result['metadata'].get('name', 'our top recommendation')
        price = top_result['metadata'].get('price', '')
        category = top_result['metadata'].get('category', 'sunglasses')
        
        price_text = f" for {price}" if price else ""
        action = _determine_action(user_query)
        
        fallback_response = {
            "response": f"Hi! I'm Yris, your sunglasses expert. Based on your search for '{user_query}', I found {len(search_results)} great options. The perfect match would be {product_name}{price_text} from our {category} collection, with a {top_result['similarity_score']:.0%} match to what you're looking for. Would you like more details about this style?",
            "action": action
        }
        return fallback_response

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        fallback_response = {
            "response": "Hi! I'm Yris, your sunglasses shopping assistant. I'm here to help you find the perfect eyewear from our premium collection. What style are you interested in today?",
            "action": "generalquery"
        }
        return fallback_response

def _generate_with_vertex_gemini(model, user_query, search_results):
    """Helper function for Vertex AI Gemini generation with real product names."""
    try:
        # Create detailed product information for the LLM
        products_info = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            product_name = metadata.get('name', f'Product {i}')
            description = metadata.get('description', 'Premium sunglasses')
            price = metadata.get('price', 'Contact for pricing')
            category = metadata.get('category', 'Sunglasses')
            style = metadata.get('style_shape', 'Classic')
            features = metadata.get('key_features', 'UV protection')
            ideal_for = metadata.get('ideal_for', 'General wear')
            
            product_info = f"""
{i}. **{product_name}**
   - Price: {price}
   - Category: {category} 
   - Style: {style}
   - Description: {description}
   - Features: {features}
   - Ideal For: {ideal_for}
   - Match Score: {result['similarity_score']:.1%}
"""
            products_info.append(product_info)

        products_text = "\n".join(products_info)

        prompt = f"""You are Yris, the sophisticated AI shopping assistant for a premium sunglasses brand. A customer searched for: "{user_query}"

Here are the matching sunglasses from our collection:
{products_text}

Instructions:
- Respond as Yris, the knowledgeable sunglasses expert
- Use ACTUAL PRODUCT NAMES (like "MinGlasses", "CatEyeGlasses", "RetroGlasses") - NEVER use generic terms like "product_1" 
- Be conversational, helpful, and mention specific product details
- Highlight what makes each product special
- Reference real prices, features, and styles from the data

Action Guidelines:
- "teleport" - Customer wants to browse a specific category/section
- "highlight" - Customer wants to see/compare specific products  
- "generalquery" - General questions about the brand/collection
- "addtocart" - Customer wants to purchase something
- "product-specific" - Customer asking about specific product details

Return valid JSON:
{{
  "response": "Your helpful response as Yris using real product names and details",
  "action": "appropriate_action"
}}

Remember: Always use the actual product names from the data, never generic product IDs!"""

        response = model.generate_content(prompt)
        
        # Clean and validate JSON response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # Parse and return as dictionary
        parsed = json.loads(response_text)
        return parsed
        
    except Exception as e:
        logger.error(f"Vertex AI Gemini generation error: {e}")
        return None

def _generate_with_gemini_api(model, user_query, search_results):
    """Helper function for Gemini API generation with real product names."""
    try:
        # Create concise product list
        products_text = ""
        for i, result in enumerate(search_results, 1):
            metadata = result.get('metadata', {})
            name = metadata.get('name', f'Product {i}')
            price = metadata.get('price', 'N/A')
            category = metadata.get('category', 'Sunglasses')
            
            products_text += f"{i}. {name} - {price} ({category}) - {result['similarity_score']:.0%} match\n"

        prompt = f"""You are Yris brand smart AI assistant for a premium sunglasses brand. Customer asked: "{user_query}"

Matching products:
{products_text}

Respond as Yris using REAL product names (never "product_X"). Be helpful and specific.

Return JSON:
{{
  "response": "Helpful response with actual product names",
  "action": "teleport|highlight|generalquery|addtocart|product-specific"
}}"""

        response = model.generate_content(prompt)
        
        # Clean and validate JSON
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        parsed = json.loads(response_text)
        return parsed
        
    except Exception as e:
        logger.error(f"Gemini API generation error: {e}")
        return None

def _determine_action(user_query):
    """Simple rule-based action determination as fallback"""
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ["what", "all", "products", "sell", "catalog", "collection", "brand"]):
        return "generalquery"
    elif any(word in query_lower for word in ["add", "cart", "buy", "purchase", "order"]):
        return "addtocart"
    elif any(word in query_lower for word in ["show", "highlight", "see", "display", "compare"]):
        return "highlight"
    elif any(word in query_lower for word in ["go to", "navigate", "section", "category", "browse"]):
        return "teleport"
    else:
        return "product-specific"

# Public API function for webhook integration
def process_query(user_query, neighbor_count=5):
    """
    Main function to process user query and return structured response.
    This is the primary function that main.py should call.
    
    Returns:
        dict: Contains 'response' and 'action' keys
    """
    try:
        # Ensure services are initialized
        if _embedding_model is None:
            initialize_services()
        
        # Get search results using real-time embeddings
        results = query_realtime_embeddings(user_query, neighbor_count)
        
        # Generate LLM response
        llm_response = generate_llm_response(user_query, results)
        
        # Add debug information if needed
        if results:
            llm_response['debug'] = {
                'products_found': len(results),
                'top_match': results[0]['metadata'].get('name', 'Unknown') if results else None,
                'similarity_score': results[0]['similarity_score'] if results else None
            }
        
        return llm_response
        
    except Exception as e:
        logger.error(f"Error processing query '{user_query}': {str(e)}")
        return {
            "response": "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment.",
            "action": "error",
            "error": str(e)
        }

def main():
    """Standalone main function for testing query functionality."""
    print("=== YRIS Sunglasses AI Assistant ===")
    
    try:
        # Initialize services
        initialize_services()
        
        user_query = input("Enter your search query: ")
        
        # Use the public API function
        response = process_query(user_query)
        
        print("\n" + "=" * 60)
        print("YRIS AI RESPONSE:")
        print("=" * 60)
        print(json.dumps(response, indent=2))
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()