import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set AWS credentials from environment variables
if os.environ.get("RACE_AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("RACE_AWS_ACCESS_KEY_ID")
if os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY")
if os.environ.get("RACE_AWS_SESSION_TOKEN"):
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get("RACE_AWS_SESSION_TOKEN")
if os.environ.get("RACE_AWS_REGION"):
    os.environ["AWS_REGION"] = os.environ.get("RACE_AWS_REGION")

import boto3
import json

instance_type = 'g6e.4xlarge'
location = 'US West (Oregon)'
operating_system = 'Linux'

def get_ec2_price(instance_type, region_display_name, os_type='Linux'):
    client = boto3.client('pricing', region_name='us-east-1')
    
    response = client.get_products(
        ServiceCode='AmazonEC2',
        Filters=[
            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_display_name},
            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': os_type},
            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
            {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
            {'Type': 'TERM_MATCH', 'Field': 'operation', 'Value': 'RunInstances'}
        ]
    )
    
    # Parse the JSON string from the price list
    price_list = response["PriceList"]
    if price_list:
        price_item = json.loads(price_list[0])
        
        # Extract pricing terms
        terms = price_item["terms"]
        
        # Get the on-demand price information
        on_demand = list(terms["OnDemand"].values())[0]
        
        # Get the price dimensions
        price_dimension = list(on_demand["priceDimensions"].values())[0]
        
        # Extract the hourly price in USD
        hourly_price = float(price_dimension["pricePerUnit"]["USD"])
        
        return hourly_price
    else:
        return None

# Example usage
hourly_price = get_ec2_price(instance_type, location, operating_system)
print(f"Hourly price for t2.micro in US East (N. Virginia): {hourly_price} USD")

# Calculate duration in hours
duration_seconds = 1800
duration_hours = duration_seconds / 3600

# Calculate total cost
total_cost = hourly_price * duration_hours

print(f"Hourly price: {hourly_price}")
print(f"Duration in hours: {duration_hours}")
print(f"Total cost: {total_cost}")
