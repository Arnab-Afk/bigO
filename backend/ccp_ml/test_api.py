"""
Test script for CCP ML API

Tests all endpoints including realtime simulation and graph generation.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_status():
    """Test status endpoint"""
    print("\n=== Testing Status Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    return data.get('initialized', False)

def test_network():
    """Test network endpoint"""
    print("\n=== Testing Network Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/network")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Nodes: {data.get('num_nodes', 0)}")
        print(f"Edges: {data.get('num_edges', 0)}")
        return True
    return False

def test_risk_scores():
    """Test risk scores endpoint"""
    print("\n=== Testing Risk Scores Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/risk/scores")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Banks with risk scores: {len(data)}")
        if data:
            print(f"Sample: {json.dumps(data[0], indent=2)}")
        return True
    return False

def test_spectral():
    """Test spectral analysis endpoint"""
    print("\n=== Testing Spectral Analysis Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/spectral")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Spectral Radius: {data.get('spectral_radius', 0)}")
        print(f"Fiedler Value: {data.get('fiedler_value', 0)}")
        print(f"Contagion Index: {data.get('contagion_index', 0)}")
        return True
    return False

def test_margins():
    """Test margins endpoint"""
    print("\n=== Testing Margins Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/margins")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Banks with margins: {len(data)}")
        if data:
            print(f"Top bank: {data[0]['bank_name']} - {data[0]['total_margin']:.4f}")
        return True
    return False

def test_stress_test():
    """Test stress test endpoint"""
    print("\n=== Testing Stress Test Endpoint ===")
    config = {
        "shock_type": "capital",
        "shock_magnitude": 0.2
    }
    response = requests.post(f"{BASE_URL}/api/stress-test", json=config)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Impact: {json.dumps(data.get('impact', {}), indent=2)}")
        return True
    return False

def test_realtime_simulation():
    """Test realtime simulation endpoints"""
    print("\n=== Testing Realtime Simulation ===")
    
    # Initialize
    print("\n1. Initializing realtime simulation...")
    response = requests.post(f"{BASE_URL}/api/realtime/init", json={
        "max_timesteps": 50
    })
    if response.status_code != 200:
        print(f"Failed to initialize: {response.status_code}")
        return False
    print(f"Initialized: {response.json()}")
    
    # Check status
    print("\n2. Checking status...")
    response = requests.get(f"{BASE_URL}/api/realtime/status")
    print(f"Status: {json.dumps(response.json(), indent=2)}")
    
    # Run some steps
    print("\n3. Running 5 simulation steps...")
    response = requests.post(f"{BASE_URL}/api/realtime/step", json={
        "n_steps": 5
    })
    if response.status_code != 200:
        print(f"Failed to run steps: {response.status_code}")
        return False
    
    data = response.json()
    print(f"Steps executed: {data.get('steps_executed', 0)}")
    if data.get('latest_state'):
        latest = data['latest_state']
        print(f"Timestep: {latest.get('timestep', 0)}")
        print(f"Defaults: {latest.get('default_count', 0)}")
        print(f"Avg Stress: {latest.get('total_stress', 0):.3f}")
    
    # Apply shock
    print("\n4. Applying liquidity shock...")
    response = requests.post(f"{BASE_URL}/api/realtime/step", json={
        "n_steps": 10,
        "shock_config": {
            "type": "liquidity",
            "magnitude": 0.3
        }
    })
    if response.status_code == 200:
        data = response.json()
        print(f"Post-shock defaults: {data.get('latest_state', {}).get('default_count', 0)}")
    
    # Get history
    print("\n5. Getting simulation history...")
    response = requests.get(f"{BASE_URL}/api/realtime/history")
    if response.status_code == 200:
        data = response.json()
        print(f"Total timesteps in history: {data.get('total_timesteps', 0)}")
    
    return True

def test_graph_generation():
    """Test graph generation endpoints"""
    print("\n=== Testing Graph Generation ===")
    
    # Check available graphs
    print("\n1. Getting available graph types...")
    response = requests.get(f"{BASE_URL}/api/graphs/available")
    if response.status_code == 200:
        data = response.json()
        print(f"Available graphs: {[g['type'] for g in data.get('graphs', [])]}")
    
    # Generate network graph
    print("\n2. Generating network graph...")
    response = requests.post(f"{BASE_URL}/api/graphs/generate", json={
        "graph_type": "network",
        "format": "plotly"
    })
    if response.status_code == 200:
        data = response.json()
        print(f"Graph type: {data.get('type', 'unknown')}")
        print(f"Graph data length: {len(data.get('data', ''))} chars")
    else:
        print(f"Failed to generate network graph: {response.status_code}")
    
    # Generate risk distribution
    print("\n3. Generating risk distribution...")
    response = requests.post(f"{BASE_URL}/api/graphs/generate", json={
        "graph_type": "risk_distribution",
        "format": "plotly"
    })
    if response.status_code == 200:
        print("Risk distribution graph generated successfully")
    
    # Generate time series (requires simulation history)
    print("\n4. Generating time series...")
    response = requests.post(f"{BASE_URL}/api/graphs/generate", json={
        "graph_type": "time_series",
        "format": "plotly"
    })
    if response.status_code == 200:
        print("Time series graph generated successfully")
    else:
        print(f"Time series requires simulation history: {response.status_code}")
    
    # Generate spectral analysis
    print("\n5. Generating spectral analysis...")
    response = requests.post(f"{BASE_URL}/api/graphs/generate", json={
        "graph_type": "spectral",
        "format": "plotly"
    })
    if response.status_code == 200:
        print("Spectral analysis graph generated successfully")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CCP ML API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Status", test_status),
        ("Network", test_network),
        ("Risk Scores", test_risk_scores),
        ("Spectral Analysis", test_spectral),
        ("Margins", test_margins),
        ("Stress Test", test_stress_test),
        ("Realtime Simulation", test_realtime_simulation),
        ("Graph Generation", test_graph_generation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    run_all_tests()
