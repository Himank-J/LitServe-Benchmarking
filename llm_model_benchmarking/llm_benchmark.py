import requests
import time
import statistics
import json
import os

class LLMBenchmark:
    def __init__(self, url="http://localhost:8000/v1/chat/completions"):
        self.url = url
        self.theoretical_max_tokens_sec = 500

    def run_benchmark(self, prompt="Tell me a story about a robot.", 
                     num_runs=5, max_new_tokens=128):
        """Run multiple benchmark iterations and collect statistics"""
        tokens_per_sec_list = []
        
        # Warmup run
        print("Warming up...")
        self._generate_and_measure(prompt, max_new_tokens)
        
        print(f"\nRunning {num_runs} benchmark iterations...")
        for i in range(num_runs):
            tokens_per_sec = self._generate_and_measure(prompt, max_new_tokens)
            tokens_per_sec_list.append(tokens_per_sec)
            print(f"Run {i+1}: {tokens_per_sec:.2f} tokens/sec")
        
        # Calculate statistics
        avg_tokens_per_sec = statistics.mean(tokens_per_sec_list)
        std_dev = statistics.stdev(tokens_per_sec_list) if len(tokens_per_sec_list) > 1 else 0
        
        print("\nBenchmark Results:")
        print(f"URL: {self.url}")
        print(f"Theoretical max tokens/sec: {self.theoretical_max_tokens_sec}")
        print(f"Average tokens/sec: {avg_tokens_per_sec:.2f} ± {std_dev:.2f}")
        print(f"Efficiency: {(avg_tokens_per_sec/self.theoretical_max_tokens_sec)*100:.1f}%")
        
        # Store results in JSON file
        results = {
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "std_dev": std_dev,
            "theoretical_max": self.theoretical_max_tokens_sec,
            "efficiency": (avg_tokens_per_sec/self.theoretical_max_tokens_sec)*100
        }
        
        self._save_results(results)
        return results

    def _generate_and_measure(self, prompt, max_new_tokens):
        """Run a single generation and measure performance"""
        # Format request payload
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.7
        }
        
        # Measure generation time
        start_time = time.perf_counter()
        
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        generation_time = end_time - start_time
        response_text = response.json()["choices"][0]["message"]["content"]
        num_new_tokens = len(response_text.split())  # Rough approximation
        tokens_per_sec = num_new_tokens / generation_time
        
        return tokens_per_sec

    def _save_results(self, results):
        """Save benchmark results to a JSON file"""
        filename = 'benchmark_results.json'
        data = {}
        
        # Load existing data if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        
        # Add timestamp as key for the new results
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        data[timestamp] = results
        
        # Write updated data back to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    benchmark = LLMBenchmark()
    benchmark.run_benchmark()