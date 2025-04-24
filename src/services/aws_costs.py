"""
AWS cost calculation utilities.
"""
import boto3
import json
import os
from dotenv import load_dotenv
from utils.logging_utils import get_logger

logger = get_logger("aws_costs")

# Load environment variables from .env file
load_dotenv()


# Mapping of AWS region codes to their display names used by the pricing API
REGION_CODE_TO_DISPLAY_NAME = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "af-south-1": "Africa (Cape Town)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ca-central-1": "Canada (Central)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-north-1": "EU (Stockholm)",
    "eu-south-1": "EU (Milan)",
    "me-south-1": "Middle East (Bahrain)",
    "sa-east-1": "South America (Sao Paulo)"
}

def get_region_display_name(region_code):
    if region_code not in REGION_CODE_TO_DISPLAY_NAME:
        raise ValueError(f"Unknown region code: {region_code}")
    
    return REGION_CODE_TO_DISPLAY_NAME[region_code]

def get_ec2_price(instance_type, region_code, os_type='Linux'):
    """
    Get the hourly price for an EC2 instance type in a specific region.
    
    Args:
        instance_type: EC2 instance type (e.g., 't2.micro', 'g6e.4xlarge')
        region_code: AWS region code (e.g., 'us-west-2')
        os_type: Operating system (default: 'Linux')
        
    Returns:
        Hourly price in USD as a float, or None if price not found
    """
    try:
        region_display_name = get_region_display_name(region_code)
        logger.debug("Getting price for %s in %s (%s)", 
                    instance_type, region_code, region_display_name)
        
        # Pricing API is only available in us-east-1
        client = boto3.client(
            'pricing', 
            region_name='us-east-1',
            aws_access_key_id=os.environ.get("RACE_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("RACE_AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("RACE_AWS_SESSION_TOKEN")
        )
        
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
        
        price_list = response["PriceList"]
        if not price_list:
            logger.warning("No price found for %s in %s", instance_type, region_code)
            return None
            
        price_item = json.loads(price_list[0])
        terms = price_item["terms"]
        on_demand = list(terms["OnDemand"].values())[0]
        price_dimension = list(on_demand["priceDimensions"].values())[0]
        hourly_price = float(price_dimension["pricePerUnit"]["USD"])
        
        logger.debug("Hourly price for %s in %s: $%f", 
                    instance_type, region_code, hourly_price)
        return hourly_price
        
    except Exception as e:
        logger.error("Error getting EC2 price: %s", str(e))
        return None

class EC2CostResult:
    def __init__(self, instance_type, region_code, os_type, hourly_price, 
                 duration_seconds, duration_hours, total_cost):
        self.instance_type = instance_type
        self.region_code = region_code
        self.region_name = REGION_CODE_TO_DISPLAY_NAME.get(region_code)
        self.os_type = os_type
        self.hourly_price = hourly_price
        self.duration_seconds = duration_seconds
        self.duration_hours = duration_hours
        self.total_cost = total_cost
    
    def __str__(self):
        return (f"EC2 {self.instance_type} in {self.region_code} ({self.region_name}): "
                f"${self.hourly_price:.4f}/hour × {self.duration_hours:.2f} hours = "
                f"${self.total_cost:.4f}")

def calculate_cost(instance_type, region_code, duration_seconds, os_type='Linux'):
    hourly_price = get_ec2_price(instance_type, region_code, os_type)
    
    if hourly_price is None:
        logger.warning("Could not calculate cost - hourly price not available")
        return None
    
    duration_hours = duration_seconds / 3600
    total_cost = hourly_price * duration_hours
    
    return EC2CostResult(
        instance_type=instance_type,
        region_code=region_code,
        os_type=os_type,
        hourly_price=hourly_price,
        duration_seconds=duration_seconds,
        duration_hours=duration_hours,
        total_cost=total_cost
    )

if __name__ == "__main__":
    # Example usage
    instance_type = 'g6e.4xlarge'
    region_code = 'us-west-2'
    duration_seconds = 1800  # 30 minutes
    
    # Calculate cost
    result = calculate_cost(instance_type, region_code, duration_seconds)
    
    if result:
        print(result)
        print(f"Instance Type: {result.instance_type}")
        print(f"Region: {result.region_code} ({result.region_name})")
        print(f"OS Type: {result.os_type}")
        print(f"Hourly Price: ${result.hourly_price:.4f}")
        print(f"Duration: {result.duration_seconds} seconds ({result.duration_hours:.2f} hours)")
        print(f"Total Cost: ${result.total_cost:.4f}")
    else:
        print("Failed to calculate cost. Check logs for details.")
