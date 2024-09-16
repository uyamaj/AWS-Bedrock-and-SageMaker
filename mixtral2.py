import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Define the model ID and prepare the payload
model_id = "mistral.mixtral-8x7b-instruct-v0:1"
user_message = "What are fun things to do in Tokyo?"

# Prepare the request body as JSON
payload = {
    "prompt": f"<s>[INST] {user_message} [/INST]",
    "max_tokens": 200,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 50
}

# Convert the payload to JSON string format
body = json.dumps(payload)

try:
    # Invoke the model using the Bedrock API
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body
    )

    # Parse the response body
    response_body = json.loads(response['body'].read())

    # # Extract the generated text
    # generated_text = response_body.get("generation", "No text generated")
    
    # Extract and print the response text.
    response_text = response["message"]["content"][0]["text"]
    print(response_text)


except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
