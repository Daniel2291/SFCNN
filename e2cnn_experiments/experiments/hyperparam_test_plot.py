import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define experiment grid
sample_sizes = [375, 750, 1500, 3000, 6000, 12000]  #sample sizes for training
N_values = [1, 2, 4, 8, 16, 24]                     #steerable filter orientations

results = {N: [] for N in N_values}

def run_exp(N, samples):
    cmd = [
        "bash", "mnist_bench_single.sh",
        "--dataset", "mnist_rot",          #using Rotated MNIST
        "--samples", str(samples),
        "--type", "regular",
        "--N", str(N),
        "--fixparams",
        "--regularize",
        "--batch_size", "32"
    ]

    print(f"\n### Running with N={N}, samples={samples} ###\n")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Extract TEST ACCURACY
    matches = re.findall(r'TEST ACCURACY\s*=\s*([0-9.]+)', result.stdout)
    if matches:
        acc = float(matches[-1])
        print(f"Test Accuracy: {acc:.4f}")
        return 1 - acc  # Convert to test error
    else:
        print(f" Could not extract accuracy for N={N}, samples={samples}")
        return None

# Run all experiments
for N in N_values:
    for samples in sample_sizes:
        error = run_exp(N, samples)
        results[N].append(error)

# Plotting
plt.figure(figsize=(8, 5))
for N, errors in results.items():
    plt.plot(sample_sizes, errors, marker='o', label=f'N={N}')

plt.xlabel('Training Samples')
plt.ylabel('Test Error')
plt.title('Test Error vs. Sample Size for Different N')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_error_vs_samples.png")
plt.show()

# Save to CSV
flat_results = [(N, s, e) for N, err_list in results.items() for s, e in zip(sample_sizes, err_list)]
df = pd.DataFrame(flat_results, columns=["N", "samples", "test_error"])
df.to_csv("test_error_results.csv", index=False)
print("\n Results saved to test_error_results.csv and plot saved to test_error_vs_samples.png")
