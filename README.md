### 1. Baseline Throughput Comparison (Images/second)

| Batch Size | Initial | Batching | Batching+Workers | Parallel Decoding | Half Precision |
|------------|---------|----------|------------------|-------------------|----------------|
| 1          | 67.38   | 66.70    | 66.54           | 66.24            | 67.29          |
| 8          | 82.28   | 82.53    | 82.56           | 82.72            | 82.79          |
| 32         | 84.54   | 84.82    | 85.00           | 85.05            | 84.89          |
| 64         | 85.18   | 85.40    | 85.45           | 85.51            | 85.53          |

### 2. API Performance Comparison (Requests/second)

| Concurrency | Initial | Batching | Batching+Workers | Parallel Decoding | Half Precision |
|-------------|---------|----------|------------------|-------------------|----------------|
| 1           | 24.12   | 21.67    | 13.46           | 10.22            | 13.51          |
| 8           | 31.32   | 32.72    | 36.58           | 24.64            | 33.41          |
| 32          | 30.20   | 34.37    | 44.09           | 37.35            | 43.03          |
| 64          | 31.06   | 33.99    | 40.62           | 39.97            | 46.71          |

### Key Observations:

1. **Baseline Throughput**:
   - All approaches show similar baseline throughput patterns
   - Marginal improvements as batch size increases
   - Half Precision shows slightly better performance at higher batch sizes

2. **API Performance**:
   - Batching+Workers shows significant improvement at higher concurrency levels
   - Half Precision performs best at highest concurrency (64), reaching 46.71 req/s
   - Initial implementation performs better at low concurrency but doesn't scale well
   - Parallel Decoding shows lower performance at low concurrency but scales reasonably well

3. **Best Performers**:
   - For low concurrency (1): Initial implementation (24.12 req/s)
   - For high concurrency (64): Half Precision (46.71 req/s)
   - Best scaling: Batching+Workers and Half Precision
