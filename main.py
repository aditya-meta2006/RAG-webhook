import base64
from flask import Flask, request, jsonify
from google.cloud import secretmanager

app = Flask(__name__)

# Initialize Secret Manager client
client = secretmanager.SecretManagerServiceClient()

# The secret path
SECRET_NAME = "projects/226903270061/secrets/webhook-basic-auth/versions/latest"

def get_basic_auth_secret():
    """Fetch the secret from Google Secret Manager."""
    response = client.access_secret_version(name=SECRET_NAME)
    secret_value = response.payload.data.decode("UTF-8")  # e.g., "metaminds_group:mmg_ai_agent"
    return secret_value

@app.route("/webhook", methods=["POST"])
def webhook():
    print("=== Incoming Request ===")

    # Print all headers
    print("\nHeaders:")
    for header, value in request.headers.items():
        print(f"{header}: {value}")

    # Print raw data
    print("\nRaw data:")
    print(request.data.decode())

    # Try to parse JSON
    try:
        req_json = request.get_json(force=True)
        print("\nParsed JSON:")
        print(req_json)
    except Exception as e:
        print("\nFailed to parse JSON:", e)
        req_json = {}

    # Basic Auth
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        print("No Authorization header received")
        return "Unauthorized", 401

    # Get secret from Secret Manager
    expected = f"Basic {base64.b64encode(get_basic_auth_secret().encode()).decode()}"

    if auth_header != expected:
        print("Authorization header mismatch")
        return "Forbidden", 403

    user_query = req_json.get("text", "No input")
    print(f"\nUser Query: {user_query}")

    return jsonify({
        "fulfillment_response": {
            "messages": [
                {"text": {"text": [f"You asked: {user_query}"]}}
            ]
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
