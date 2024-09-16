import streamlit as st
import boto3
from botocore.exceptions import ClientError

# Streamlit app title
st.title("Ask Mixtral: Your Travel Buddy 🌍✈️")

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Function to call Mixtral model
def get_mixtral_response(user_message):
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]
    
    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={"maxTokens": 800, "temperature": 0.7, "topP": 0.7},
            additionalModelRequestFields={"top_k": 50}
        )
        # Extract the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except (ClientError, Exception) as e:
        st.error(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None

# Input from the user
user_input = st.text_input("Ask a question about Tokyo or anything else:")

# When the button is pressed, call the API
if st.button("Ask Mixtral 8x7B"):
    if user_input:
        with st.spinner("Getting a response from Mixtral 8x7B..."):
            response = get_mixtral_response(f"<s>[INST]{user_input}[/INST]")
            if response:
                st.success("Response:")
                st.write(response)
            else:
                st.error("Failed to get a response.")
