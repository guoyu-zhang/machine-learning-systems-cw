import os, sys
import requests
import time
import concurrent.futures
import numpy as np

URL = "http://127.0.0.1:8000/rag"

payload = {"query": "Tell me about a movie with time travel", "k": 3}

def send_request():
    start_time = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=10)
        end_time = time.time()
        latency = round(end_time - start_time, 3)
        return response.status_code, latency
    except requests.exceptions.RequestException:
        return 0, None

def run_load_test(concurrent_requests=10, total_requests=50):
    print(f"Starting load test with {concurrent_requests} concurrent requests and {total_requests} total requests.")

    latencies = []
    errors = 0

    start_overall = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request) for _ in range(total_requests)]

        for future in concurrent.futures.as_completed(futures):
            status, latency = future.result()
            if status != 200 or latency is None:
                errors += 1
            else:
                latencies.append(latency)

    end_overall = time.time()
    test_duration = end_overall - start_overall

    if latencies:
        latencies_np = np.array(latencies)
        print("\n--- Load Test Results ---")
        print(f"Total Requests: {total_requests}")
        print(f"Total Errors: {errors} ({(errors/total_requests)*100:.2f}%)")
        print(f"Total Duration: {test_duration:.2f}s")
        print(f"Throughput: {total_requests / test_duration:.2f} requests/sec")
        print(f"Average Latency: {latencies_np.mean():.3f}s")
        print(f"Median Latency: {np.median(latencies_np):.3f}s")
        print(f"95th Percentile Latency: {np.percentile(latencies_np, 95):.3f}s")
        print(f"Max Latency: {latencies_np.max():.3f}s")
    else:
        print("No successful responses.")

if __name__ == "__main__":
    run_load_test(concurrent_requests=5, total_requests=50)
