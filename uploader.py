import os
import json
import time
import pandas as pd
from google.cloud import storage
import tempfile
from vertexai.language_models import TextEmbeddingModel
import vertexai

# --- Configuration ---
PROJECT_ID = "prj-d-meta-playground-a917"
LOCATION = "us-central1"
SOURCE_BUCKET_NAME = "versai_yris_brand_data"
UPLOAD_BUCKET_NAME = "vertex-ai-yris-brand-data-embeddings"

# Files in your GCS bucket
PRODUCT_FILE = "sunglasses_products_csv.xlsx"
DESCRIPTION_FILE = "sunglasses_description.txt"

# --- Initialize Vertex AI SDK ---
vertexai.init(project=PROJECT_ID, location=LOCATION)

def clean_old_files_from_gcs():
    """Clean old files from GCS bucket to avoid conflicts."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(UPLOAD_BUCKET_NAME)
        
        # List and delete old files
        blobs = bucket.list_blobs(prefix="embeddings_folder_")
        deleted_count = 0
        for blob in blobs:
            blob.delete()
            deleted_count += 1
        
        if deleted_count > 0:
            print(f"Cleaned {deleted_count} old files from GCS bucket")
        
    except Exception as e:
        print(f"Warning: Could not clean old files: {str(e)}")

def download_file_from_gcs(bucket_name, object_name):
    """Download a file from Google Cloud Storage to a temporary location."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        
        # Create temp file with same extension
        _, ext = os.path.splitext(object_name)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        local_filename = temp_file.name
        temp_file.close()
        
        blob.download_to_filename(local_filename)
        print(f"Downloaded {object_name}")
        return local_filename
        
    except Exception as e:
        print(f"Error downloading {object_name}: {str(e)}")
        return None

def load_product_data():
    """Load product data from GCS."""
    print(f"Loading product data: {PRODUCT_FILE}")
    local_path = download_file_from_gcs(SOURCE_BUCKET_NAME, PRODUCT_FILE)
    
    if local_path is None:
        return None
    
    try:
        df = pd.read_excel(local_path)
        print(f"Product data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        os.unlink(local_path)  # Clean up temp file
        return df
        
    except Exception as e:
        print(f"Error loading product file: {str(e)}")
        if os.path.exists(local_path):
            os.unlink(local_path)
        return None

def load_description_data():
    """Load description data from GCS."""
    print(f"Loading description data: {DESCRIPTION_FILE}")
    local_path = download_file_from_gcs(SOURCE_BUCKET_NAME, DESCRIPTION_FILE)
    
    if local_path is None:
        return ""
    
    try:
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Description data loaded: {len(content)} characters")
        os.unlink(local_path)  # Clean up temp file
        return content
        
    except Exception as e:
        print(f"Error loading description file: {str(e)}")
        if os.path.exists(local_path):
            os.unlink(local_path)
        return ""

def extract_sunglasses_metadata(row, row_index):
    """Extract metadata specifically for sunglasses product structure."""
    metadata = {}
    
    # Map your specific column structure
    field_mappings = {
        'Product Name': 'name',
        'Category': 'category', 
        'Description': 'description',
        'Frame Material': 'frame_material',
        'Lens Type': 'lens_type',
        'UV Protection': 'uv_protection',
        'Key Features': 'key_features',
        'Price': 'price',
        'Known For': 'known_for',
        'Style/Shape': 'style_shape',
        'Ideal For': 'ideal_for',
        'How to Use': 'how_to_use',
        'Packaging': 'packaging',
        'Availability': 'availability',
        'Guarantee': 'guarantee'
    }
    
    # Extract and clean product data
    for field_name in row._fields:
        value = getattr(row, field_name)
        
        if pd.isna(value) or str(value).strip() == '':
            continue
        
        clean_value = str(value).strip()
        
        # Map to standardized field names
        if field_name in field_mappings:
            standardized_field = field_mappings[field_name]
            metadata[standardized_field] = clean_value
        else:
            # Keep unmapped fields with cleaned names
            clean_field_name = field_name.replace('_', ' ').replace('/', ' ').title()
            metadata[clean_field_name] = clean_value
    
    # Create product ID from product name or use fallback
    product_name = metadata.get('name', f'Product_{row_index + 1}')
    product_id = product_name.lower().replace(' ', '_').replace('glasses', '').replace('sunglasses', '').strip('_')
    
    # Ensure we have essential fields
    if 'category' not in metadata:
        metadata['category'] = 'Sunglasses'
    
    if 'brand' not in metadata:
        metadata['brand'] = 'Sunglasses'
    
    return product_id, metadata

def generate_comprehensive_embeddings(product_df, description_text):
    """Generate embeddings with rich metadata for sunglasses products."""
    print("Generating comprehensive embeddings for sunglasses products...")
    model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
    
    # Extract brand context from description
    brand_context = ""
    if description_text:
        # Extract key brand information
        lines = description_text.split('\n')
        brand_essence_lines = []
        mission_line = ""
        
        for line in lines:
            if line.strip():
                if 'Brand Essence:' in line or 'Mission:' in line:
                    continue
                elif any(keyword in line for keyword in ['Style-Driven', 'Quality', 'Protection', 'fashion-forward', 'premium']):
                    brand_essence_lines.append(line.strip())
                elif 'Mission:' in line:
                    mission_line = line.replace('Mission:', '').strip()
        
        if mission_line:
            brand_context = mission_line
        elif brand_essence_lines:
            brand_context = ' '.join(brand_essence_lines[:2])
        else:
            brand_context = "Premium fashion eyewear brand offering stylish, protective sunglasses."
    
    embeddings_data = {}
    
    for i, row in enumerate(product_df.itertuples(index=False)):
        try:
            # Extract metadata using sunglasses-specific logic
            product_id, metadata = extract_sunglasses_metadata(row, i)
            
            # Create comprehensive text for embedding
            text_parts = []
            
            # Primary product information
            if metadata.get('name'):
                text_parts.append(f"Product: {metadata['name']}")
            
            if metadata.get('description'):
                text_parts.append(f"Description: {metadata['description']}")
            
            if metadata.get('category'):
                text_parts.append(f"Category: {metadata['category']}")
            
            if metadata.get('style_shape'):
                text_parts.append(f"Style: {metadata['style_shape']}")
            
            # Technical specifications
            if metadata.get('frame_material'):
                text_parts.append(f"Frame: {metadata['frame_material']}")
            
            if metadata.get('lens_type'):
                text_parts.append(f"Lens: {metadata['lens_type']}")
            
            if metadata.get('uv_protection'):
                text_parts.append(f"UV Protection: {metadata['uv_protection']}")
            
            if metadata.get('key_features'):
                text_parts.append(f"Features: {metadata['key_features']}")
            
            # Price and positioning
            if metadata.get('price'):
                text_parts.append(f"Price: {metadata['price']}")
            
            if metadata.get('known_for'):
                text_parts.append(f"Known For: {metadata['known_for']}")
            
            # Usage and target audience
            if metadata.get('ideal_for'):
                text_parts.append(f"Ideal For: {metadata['ideal_for']}")
            
            # Add brand context
            if brand_context:
                text_parts.append(f"Brand: {brand_context}")
            
            # Create final embedding text
            embedding_text = ". ".join(text_parts)
            
            if not embedding_text.strip():
                print(f"Skipping product {i+1}: No content for embedding")
                continue
            
            # Generate embedding
            embedding_vector = model.get_embeddings([embedding_text])[0].values
            
            # Store in the format expected by query system
            embeddings_data[product_id] = {
                "embedding": embedding_vector,
                "metadata": metadata
            }
            
            if (i + 1) % 3 == 0:
                print(f"Processed {i + 1}/{len(product_df)} products")
            
            # Print first product for verification
            if i == 0:
                print(f"\nFirst product processed:")
                print(f"ID: {product_id}")
                print(f"Name: {metadata.get('name', 'N/A')}")
                print(f"Category: {metadata.get('category', 'N/A')}")
                print(f"Price: {metadata.get('price', 'N/A')}")
                print(f"Embedding text preview: {embedding_text[:200]}...")
                
        except Exception as e:
            print(f"Error processing product {i+1}: {str(e)}")
            continue

    print(f"\nGenerated embeddings for {len(embeddings_data)} products")
    return embeddings_data

def upload_embeddings_to_gcs(embeddings_data):
    """Upload embeddings with metadata to GCS."""
    try:
        # Create unique folder name
        timestamp = int(time.time())
        gcs_folder = f"embeddings_folder_{timestamp}"
        gcs_file_name = "product_embeddings.json"
        
        # Create JSON content with proper structure
        json_content = json.dumps(embeddings_data, indent=2, ensure_ascii=False)
        
        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(UPLOAD_BUCKET_NAME)
        blob = bucket.blob(f"{gcs_folder}/{gcs_file_name}")
        
        blob.upload_from_string(json_content, content_type='application/json')
        
        gcs_file_uri = f"gs://{UPLOAD_BUCKET_NAME}/{gcs_folder}/{gcs_file_name}"
        print(f"Uploaded embeddings to: {gcs_file_uri}")
        
        return {
            "gcs_uri": gcs_file_uri,
            "bucket": UPLOAD_BUCKET_NAME,
            "folder": gcs_folder,
            "file": gcs_file_name,
            "total_products": len(embeddings_data)
        }
        
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return None

def main():
    """Main function that generates and uploads embeddings."""
    print("=== Sunglasses Embeddings Generator ===")
    
    try:
        # Clean old files first
        print("\nStep 1: Cleaning old files...")
        clean_old_files_from_gcs()
        
        # Load data
        print("\nStep 2: Loading data from GCS...")
        product_df = load_product_data()
        if product_df is None:
            print("Failed to load product data. Exiting...")
            return
        
        description_text = load_description_data()
        
        print(f"\nData loaded:")
        print(f"- Products: {len(product_df)} items")
        print(f"- Description: {len(description_text)} characters")
        
        # Generate embeddings with rich metadata
        print("\nStep 3: Generating embeddings with sunglasses metadata...")
        embeddings_data = generate_comprehensive_embeddings(product_df, description_text)
        
        if not embeddings_data:
            print("No embeddings generated. Exiting...")
            return
        
        # Upload to GCS (no local backup)
        print("\nStep 4: Uploading embeddings to GCS...")
        upload_result = upload_embeddings_to_gcs(embeddings_data)
        
        if upload_result:
            print("\n" + "="*60)
            print("SUCCESS! Sunglasses embeddings generated and uploaded!")
            print("="*60)
            print(f"GCS Location: {upload_result['gcs_uri']}")
            print(f"Total Products: {upload_result['total_products']}")
            
            # Show product samples
            print(f"\nProduct samples:")
            for i, (product_id, data) in enumerate(list(embeddings_data.items())[:3]):
                metadata = data['metadata']
                print(f"{i+1}. {metadata.get('name', product_id)} - {metadata.get('price', 'N/A')}")
                print(f"   Category: {metadata.get('category', 'N/A')}")
                print(f"   Style: {metadata.get('style_shape', 'N/A')}")
            
            # Create configuration for query system
            config = {
                "project_id": PROJECT_ID,
                "location": LOCATION,
                "gcs_bucket": upload_result['bucket'],
                "gcs_folder": upload_result['folder'],
                "gcs_file": upload_result['file'],
                "gcs_uri": upload_result['gcs_uri'],
                "total_products": upload_result['total_products'],
                "embedding_model": "text-multilingual-embedding-002",
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save configuration
            config_filename = "embeddings_config.json"
            with open(config_filename, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to: {config_filename}")
            
        else:
            print("Failed to upload embeddings.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()