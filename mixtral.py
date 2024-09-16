# Use the Converse API to send a text message to Mixtral 8x7B Instruct.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Titan Text Premier.
model_id = "mistral.mixtral-8x7b-instruct-v0:1"

# Start a conversation with the user message.
user_message = """<s>[INST]What are fun things to do in Tokyo?[/INST]"""
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId="mistral.mixtral-8x7b-instruct-v0:1",
        messages=conversation,
        inferenceConfig={"maxTokens":800,"temperature":0.7,"topP":0.7},
        additionalModelRequestFields={"top_k":50}
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
