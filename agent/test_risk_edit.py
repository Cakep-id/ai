import requests
import json

# Test get pending reports
try:
    response = requests.get("http://localhost:8000/api/admin/reports/pending")
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    print()
    
    # Test update risk level jika ada reports
    reports = response.json().get('reports', [])
    if reports:
        report_id = reports[0]['report_id']
        print(f"Testing update risk level for report {report_id}")
        
        update_response = requests.put(
            f"http://localhost:8000/api/admin/update-risk-level/{report_id}",
            json={"new_risk_level": "MEDIUM", "notes": "Test dari admin"}
        )
        print("Update Status Code:", update_response.status_code)
        print("Update Response:", update_response.json())
    else:
        print("No reports found to test update")
        
except Exception as e:
    print(f"Error: {e}")
