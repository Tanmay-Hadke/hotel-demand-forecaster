import json
import boto3
import datetime

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('HotelBookings')

# Hardcoded from your training stats (or fetch from a config DB)
MEAN_ROOMS = 150.5 
STD_ROOMS = 45.2

def lambda_handler(event, context):
    hotel_id = event.get('hotel_id', 'CITY_HOTEL_1')
    target_date = event.get('target_date', str(datetime.date.today()))
    
    # In a real app, query DynamoDB for the 30 days prior to target_date
    # response = table.query(...) 
    # For this blueprint, we simulate the retrieved 30-day historical window
    
    features = []
    # Simulating data processing loop
    for i in range(30):
        # 1. Get raw rooms (simulated)
        raw_rooms = 160 
        
        # 2. Normalize rooms
        norm_rooms = (raw_rooms - MEAN_ROOMS) / STD_ROOMS
        
        # 3. Extract and scale temporal features
        # day_of_week scaled to [-1, 1], month scaled to [-1, 1]
        day_of_week = 0.5 
        month = -0.2      
        
        features.append([norm_rooms, day_of_week, month])
        
    return {
        "statusCode": 200,
        "hotel_id": hotel_id,
        "target_date": target_date,
        "features": features # Shape: (30, 3)
    }