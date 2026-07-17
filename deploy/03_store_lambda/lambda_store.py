import json
import boto3
import decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ForecastedDemand')

MEAN_ROOMS = 150.5 
STD_ROOMS = 45.2

def lambda_handler(event, context):
    normalized_pred = event.get('normalized_prediction')
    hotel_id = event.get('hotel_id')
    target_date = event.get('target_date')
    
    # Inverse transform to actual rooms
    actual_rooms = (normalized_pred * STD_ROOMS) + MEAN_ROOMS
    
    # DynamoDB requires floats to be cast to Decimals
    rooms_decimal = decimal.Decimal(str(round(actual_rooms, 2)))
    
    # Write to database
    table.put_item(
        Item={
            'hotel_id': hotel_id,
            'forecast_date': target_date,
            'predicted_demand': rooms_decimal,
            'status': 'READY_FOR_REVENUE_MANAGER'
        }
    )
    
    return {
        "statusCode": 200,
        "message": f"Successfully forecasted {rooms_decimal} rooms for {target_date}"
    }