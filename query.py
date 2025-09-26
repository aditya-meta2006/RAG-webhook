import os
import json
import vertexai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
from google.cloud import storage

# ==============================
# Load Configuration from JSON
# ==============================
CONFIG_PATH = "embeddings_config.json"

def load_config():
    """Load configuration from embeddings config file."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            return config
        else:
            print(f"Config file {CONFIG_PATH} not found. Please run uploader.py first.")
            return None
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

# Load config
config = load_config()
if not config:
    exit(1)

PROJECT_ID = config["project_id"]
LOCATION = config["location"]

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

def load_embeddings_from_gcs():
    """
    Load product embeddings directly from Google Cloud Storage using config.
    """
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(config["gcs_bucket"])
        
        # Construct the full path
        gcs_file_path = f"{config['gcs_folder']}/{config['gcs_file']}"
        blob = bucket.blob(gcs_file_path)
        
        # Download the JSON content
        embeddings_data = json.loads(blob.download_as_text())
        
        print(f"Successfully loaded {len(embeddings_data)} sunglasses products from GCS")
        
        # Print sample product info
        if embeddings_data:
            sample_id = list(embeddings_data.keys())[0]
            sample_metadata = embeddings_data[sample_id].get('metadata', {})
            print(f"Sample: {sample_metadata.get('name', sample_id)} - {sample_metadata.get('price', 'N/A')}")
        
        return embeddings_data
        
    except Exception as e:
        print(f"Error loading embeddings from GCS: {e}")
        return {}

def query_realtime_embeddings(user_query, neighbor_count=5):
    """
    Query sunglasses products using real-time embedding computation and cosine similarity.
    """
    try:
        if not user_query or not user_query.strip():
            print("Please provide a valid search query")
            return []

        if neighbor_count < 1:
            neighbor_count = 5

        # Load stored product embeddings from GCS
        product_embeddings = load_embeddings_from_gcs()
        
        if not product_embeddings:
            print("No product embeddings available")
            return []

        # Generate embedding for user query
        model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
        query_embedding = model.get_embeddings([user_query])[0].values
        
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
                print(f"Error processing product {product_id}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top N results
        return similarities[:neighbor_count]

    except Exception as e:
        print(f"Error querying real-time embeddings: {str(e)}")
        return []

def format_product_info(result):
    """Format product information for LLM context."""
    metadata = result.get('metadata', {})
    
    info_parts = []
    
    # Essential product info
    name = metadata.get('name', 'Unknown Product')
    info_parts.append(f"Name: {name}")
    
    if metadata.get('price'):
        info_parts.append(f"Price: {metadata['price']}")
    
    if metadata.get('category'):
        info_parts.append(f"Category: {metadata['category']}")
    
    if metadata.get('description'):
        info_parts.append(f"Description: {metadata['description']}")
    
    if metadata.get('style_shape'):
        info_parts.append(f"Style: {metadata['style_shape']}")
    
    if metadata.get('key_features'):
        info_parts.append(f"Features: {metadata['key_features']}")
    
    if metadata.get('ideal_for'):
        info_parts.append(f"Ideal For: {metadata['ideal_for']}")
    
    return " | ".join(info_parts)

def generate_llm_response(user_query, search_results):
    """
    Generate a natural language response using an LLM with actual product names.
    """
    try:
        if not search_results:
            fallback_response = {
                "response": "I'm Yris, your AI shopping assistant! I couldn't find specific matches for that query, but I'd be happy to help you explore our sunglasses collection. We offer styles ranging from classic to trendy, all with premium UV protection. What type of look are you going for?",
                "action": "generalquery"
            }
            return json.dumps(fallback_response, indent=2)

        # Try Vertex AI Gemini models first
        try:
            model_versions = ["gemini-2.5-flash"]
            for version in model_versions:
                try:
                    model = GenerativeModel(version)
                    response_text = _generate_with_vertex_gemini(model, user_query, search_results)
                    if response_text:
                        return response_text
                except Exception as e:
                    print(f"Failed with {version}: {e}")
                    continue
        except Exception as e:
            print(f"Vertex AI Gemini attempt failed: {e}")

        # Try Gemini API as backup
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                response_text = _generate_with_gemini_api(model, user_query, search_results)
                if response_text:
                    return response_text
        except Exception as e:
            print(f"Gemini API attempt failed: {e}")

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
        return json.dumps(fallback_response, indent=2)

    except Exception as e:
        print(f"Error generating LLM response: {str(e)}")
        fallback_response = {
            "response": "Hi! I'm Yris, your sunglasses shopping assistant. I'm here to help you find the perfect eyewear from our premium collection. What style are you interested in today?",
            "action": "generalquery"
        }
        return json.dumps(fallback_response, indent=2)

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
        
        # Validate JSON
        parsed = json.loads(response_text)
        return json.dumps(parsed, indent=2)
        
    except Exception as e:
        print(f"Vertex AI Gemini generation error: {e}")
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
        return json.dumps(parsed, indent=2)
        
    except Exception as e:
        print(f"Gemini API generation error: {e}")
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

def main():
    """Standalone main function for testing query functionality."""
    print("=== YRIS Sunglasses AI Assistant ===")
    
    user_query = input("Enter your search query: ")

    # Get search results using real-time embeddings
    results = query_realtime_embeddings(user_query)

    if results:
        print("\n" + "=" * 60)
        print("YRIS AI RESPONSE:")
        print("=" * 60)
        llm_response = generate_llm_response(user_query, results)
        print(llm_response)
        print("=" * 60)

        print(f"\nDebug - Product Results:")
        for i, result in enumerate(results, start=1):
            metadata = result.get('metadata', {})
            product_name = metadata.get('name', result['product_id'])
            price = metadata.get('price', 'N/A')
            category = metadata.get('category', 'N/A')
            print(f"{i}. {product_name}")
            print(f"   Price: {price}")
            print(f"   Category: {category}")
            print(f"   Similarity Score: {result['similarity_score']:.4f}")
            print(f"   Style: {metadata.get('style_shape', 'N/A')}")
            print()
    else:
        print("No matching results found.")

if __name__ == "__main__":
    main()