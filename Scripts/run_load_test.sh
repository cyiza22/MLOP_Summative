#!/bin/bash

echo "Starting Load Test with Locust"
echo "================================"
echo ""
echo "Test Configuration:"
echo "  - Target: http://localhost:5000"
echo "  - Users: 100"
echo "  - Spawn Rate: 10 users/second"
echo "  - Duration: 60 seconds"
echo ""

# Run locust with different user counts
for users in 10 50 100; do
    echo "Running test with $users users..."
    locust -f locustfile.py \
        --host=http://localhost:5000 \
        --users=$users \
        --spawn-rate=10 \
        --run-time=60s \
        --headless \
        --html=results/locust_report_${users}_users.html \
        --csv=results/locust_${users}_users
    
    echo "Test with $users users completed"
    echo "---"
    sleep 5
done

echo ""
echo "All load tests completed!"
echo "Results saved in results/ directory"