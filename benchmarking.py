import base64
import concurrent.futures
import time
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import psutil
try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

print('GPU_AVAILABLE', GPU_AVAILABLE)

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# Constants
SERVER_URL =  "http://localhost:8000/predict"
TEST_IMAGE_URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/1272/2280/testSample/testSample/img_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241115%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241115T154043Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=58e5c79ba897aa871408b7d061a84aafde1fbc1cce780ead1a50c8011547f311e6346b1da264d937f9ba4fc8ce46c902ef0021784df75cdee71954c32b7c5fd9637d20d40f4dfc0392977b9aae98b568412f073fe9480a8f8f851986e504e68907f62f0922829b34c79b522ec65ff21af64ef01e0e4c2f0b88e20d4ca3bf63bb3b4557276179454326a801a39ff443b781a4f956375c61b0240a28d8336989d445b0b7e266d2cd799afcaf3c5c033454bc4ba7b3f38f0c1b881bf293bdfe22028ad25d9f263d43597004fb9bb2fa5f363c2eae5976487d629f7d75a018b43e14abd552e02dfe36204508e4f6fdae7179ad603c095d6ce4298fe2c124643375ba'

def get_baseline_throughput(batch_size, num_iterations=10):
    """Calculate baseline model throughput without API overhead"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Create model and move to device
    state_dict = torch.load('model_checkpoint/model.pt', map_location=device)
    model = MNISTModel()  
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create random input data
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    throughputs = []
    
    # Warm-up run
    with torch.no_grad():
        model(x)
    
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # Each request processes batch_size images
        reqs_per_sec = (batch_size)/(t1-t0)  # Requests per second
        throughputs.append(reqs_per_sec)
    
    return np.mean(throughputs)  # reqs/sec

def prepare_test_payload():
    """Prepare a test image payload"""
    img_data = urlopen(TEST_IMAGE_URL).read()
    return base64.b64encode(img_data).decode('utf-8')

def send_request(payload):
    """Send a single request and measure response time"""
    start_time = time.time()
    response = requests.post(SERVER_URL, json={"image": payload})
    end_time = time.time()
    return end_time - start_time, response.status_code

def get_system_metrics():
    """Get current GPU and CPU usage"""
    metrics = {"cpu_usage": psutil.cpu_percent(0.1)}
    if GPU_AVAILABLE:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            metrics["gpu_usage"] = sum([gpu.utilization for gpu in gpu_stats.gpus])
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            metrics["gpu_usage"] = -1
    else:
        print("CUDA not available - GPU processing disabled")
        metrics["gpu_usage"] = -1
    return metrics

def benchmark_api(num_requests=100, concurrency_level=10):
    """Benchmark the API server"""
    payload = prepare_test_payload()
    system_metrics = []
    
    start_benchmark_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(num_requests)]
        response_times = []
        status_codes = []
        
        # Collect system metrics during the benchmark
        while any(not f.done() for f in futures):
            system_metrics.append(get_system_metrics())
            time.sleep(0.1)
        
        for future in futures:
            response_time, status_code = future.result()
            response_times.append(response_time)
            status_codes.append(status_code)
    
    end_benchmark_time = time.time()
    total_benchmark_time = end_benchmark_time - start_benchmark_time
    
    avg_cpu = np.mean([m["cpu_usage"] for m in system_metrics])
    avg_gpu = np.mean([m["gpu_usage"] for m in system_metrics]) if GPU_AVAILABLE else -1
    
    return {
        "total_requests": num_requests,
        "concurrency_level": concurrency_level,
        "total_time": total_benchmark_time,
        "avg_response_time": np.mean(response_times) * 1000,  # Convert to ms
        "success_rate": (status_codes.count(200) / num_requests) * 100,
        "requests_per_second": num_requests / total_benchmark_time,
        "avg_cpu_usage": avg_cpu,
        "avg_gpu_usage": avg_gpu
    }

def run_benchmarks():
    """Run comprehensive benchmarks and create plots"""
    # Test different batch sizes for baseline throughput
    batch_sizes = [1, 8, 32, 64]
    baseline_throughput = []
    
    print("Running baseline throughput tests...")
    for batch_size in batch_sizes:
        reqs_per_sec = get_baseline_throughput(batch_size)
        baseline_throughput.append(reqs_per_sec)
        print(f"Batch size {batch_size}: {reqs_per_sec:.2f} reqs/sec")
    
    # Test different concurrency levels for API
    concurrency_levels = [1, 8, 32, 64]
    api_throughput = []
    cpu_usage = []
    gpu_usage = []
    
    print("\\nRunning API benchmarks...")
    for concurrency in concurrency_levels:
        metrics = benchmark_api(num_requests=128, concurrency_level=concurrency)
        api_throughput.append(metrics["requests_per_second"])
        cpu_usage.append(metrics["avg_cpu_usage"])
        gpu_usage.append(metrics["avg_gpu_usage"])
        print(f"Concurrency {concurrency}: {metrics['requests_per_second']:.2f} reqs/sec, "
              f"CPU: {metrics['avg_cpu_usage']:.1f}%, GPU: {metrics['avg_gpu_usage']:.1f}%")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Throughput comparison
    plt.subplot(1, 3, 1)
    plt.plot(batch_sizes, baseline_throughput, 'b-', label='Baseline Model')
    plt.plot(concurrency_levels, api_throughput, 'r-', label='API Server')
    plt.xlabel('Batch Size / Concurrency Level')
    plt.ylabel('Throughput (requests/second)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.grid(True)
    
    # CPU Usage
    plt.subplot(1, 3, 2)
    plt.plot(concurrency_levels, cpu_usage, 'g-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage')
    plt.grid(True)
    
    # GPU Usage
    plt.subplot(1, 3, 3)
    plt.plot(concurrency_levels, gpu_usage, 'm-')
    plt.xlabel('Concurrency Level')
    plt.ylabel('GPU Usage (%)')
    plt.title('GPU Usage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    run_benchmarks() 