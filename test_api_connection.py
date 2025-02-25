import os
import requests
import json
from dotenv import load_dotenv

def main():
    """Test Hyperbolic API connection."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("HYPERBOLIC_API_KEY")
    if not api_key:
        print("ERROR: HYPERBOLIC_API_KEY environment variable is not set")
        return
    
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Test marketplace endpoint - requires POST method
    marketplace_endpoint = "https://api.hyperbolic.xyz/v1/marketplace"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use POST request with empty filters
    try:
        print(f"\nTesting marketplace endpoint: {marketplace_endpoint}")
        response = requests.post(marketplace_endpoint, headers=headers, json={"filters": {}})
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Marketplace API connection successful!")
            # Check if we have data
            data = response.json()
            print(f"Found {len(data.get('instances', []))} instances")
            
            # Count available GPUs
            available_gpus = 0
            available_instances = []
            for instance in data.get('instances', []):
                if not instance.get('reserved', True):
                    available_gpus += instance.get('gpus_total', 0) - instance.get('gpus_reserved', 0)
                    available_instances.append(instance)
            
            print(f"Available GPUs: {available_gpus}")
            
            # Show a few example instances
            print("\nExample available instances:")
            count = 0
            for instance in available_instances[:3]:  # Show at most 3
                gpu_model = instance.get('hardware', {}).get('gpus', [{}])[0].get('model', 'Unknown')
                price = instance.get('pricing', {}).get('price', {}).get('amount', 0) / 100
                print(f"- Cluster: {instance.get('cluster_name')}")
                print(f"  Node: {instance.get('id')}")
                print(f"  GPU: {gpu_model}")
                print(f"  Price: ${price}/hour")
                print(f"  Available: {instance.get('gpus_total', 0) - instance.get('gpus_reserved', 0)} GPUs")
                print()
                count += 1
            
            # Save instance info for rent_compute test
            if available_instances:
                test_instance = available_instances[0]
                test_rent_compute_validity(
                    api_key, 
                    test_instance.get('cluster_name', ''),
                    test_instance.get('id', '')
                )
        else:
            print(f"Marketplace API connection failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error response: {response.text}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test instances endpoint - requires GET method
    instances_endpoint = "https://api.hyperbolic.xyz/v1/marketplace/instances"
    try:
        print(f"\nTesting instances endpoint: {instances_endpoint}")
        response = requests.get(instances_endpoint, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Instances API connection successful!")
            data = response.json()
            print(f"Active instances: {len(data.get('instances', []))}")
            
            if data.get('instances'):
                print("\nActive instances:")
                for instance in data.get('instances', [])[:3]:  # Show at most 3
                    print(f"- ID: {instance.get('id')}")
                    print(f"  Status: {instance.get('status')}")
                    print(f"  Created: {instance.get('created_at')}")
                    print()
            else:
                print("No active instances found")
        else:
            print(f"Instances API connection failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error response: {response.text}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

def test_rent_compute_validity(api_key, cluster_name, node_name):
    """Test if the rent compute endpoint accepts our API key but don't actually rent anything."""
    if not cluster_name or not node_name:
        print("\nSkipping rent_compute test due to lack of available instances")
        return
        
    print(f"\nTesting rent_compute API validity (without actually renting)")
    print(f"Using cluster: {cluster_name}, node: {node_name}")
    
    # We'll make a request that should fail due to invalid parameters,
    # but still tell us if the API key is valid
    
    # Use an invalid GPU count to prevent actual rental
    gpu_count = -1  
    
    endpoint = "https://api.hyperbolic.xyz/v1/marketplace/instances/create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "cluster_name": cluster_name,
        "node_name": node_name,
        "gpu_count": gpu_count,
        "instance_type": "gpu"
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        print(f"Response status code: {response.status_code}")
        
        # A 400 error is expected due to invalid params, but indicates API key is valid
        # A 401 error indicates invalid API key
        if response.status_code == 400:
            print("API key accepted but request failed due to invalid parameters (as expected)")
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error response: {response.text}")
        elif response.status_code == 401:
            print("API key validation failed")
        else:
            print(f"Unexpected response: {response.status_code}")
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            except:
                print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 