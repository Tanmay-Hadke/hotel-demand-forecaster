import json
import boto3
import torch
import os
from model import AttentionLSTM

s3 = boto3.client('s3')
BUCKET_NAME = 'siteminder-ml-models'
MODEL_KEY = 'best_attention_lstm.pt'
MODEL_PATH = '/tmp/model.pt'

# Initialize model globally to survive warm starts
model = None

def load_model():
    global model
    if model is None:
        # Download from S3 to Lambda's /tmp/ storage
        s3.download_file(BUCKET_NAME, MODEL_KEY, MODEL_PATH)
        model = AttentionLSTM(input_size=3, hidden_size=128)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    return model

def lambda_handler(event, context):
    payload = event.get('features') # Passed from Data Prep Lambda
    
    # Convert list back to PyTorch Tensor with batch dimension (1, 30, 3)
    input_tensor = torch.tensor(payload, dtype=torch.float32).unsqueeze(0)
    
    model = load_model()
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
        
    return {
        "statusCode": 200,
        "hotel_id": event.get('hotel_id'),
        "target_date": event.get('target_date'),
        "normalized_prediction": prediction
    }