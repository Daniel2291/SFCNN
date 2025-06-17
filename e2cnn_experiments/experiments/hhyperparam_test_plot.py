import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define experiment grid
sample_sizes = [375, 750, 1500, 3000, 6000, 12000] #sample sizes for training
#sample_sizes = [375, 750]
N_values = [1, 2, 4, 8, 16, 24]                    #steerable filter orientations                
#N_values = [1, 2]

results = {samples: [] for samples in sample_sizes}

def run_exp(N, samples):
    cmd = [
        "bash", "mnist_bench_single.sh",
        "--dataset", "mnist_rot",
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
        return (1 - acc) * 100  # Convert to test error in percent
    else:
        print(f" Could not extract accuracy for N={N}, samples={samples}")
        return None

# Run all experiments
for samples in sample_sizes:
    for N in N_values:
        error = run_exp(N, samples)
        results[samples].append(error)

# Plotting
plt.figure(figsize=(8, 5))
for samples, errors in results.items():
    plt.plot(N_values, errors, marker='o', label=f'Samples={samples}')

plt.xlabel('N')
plt.ylabel('Test Error[%]')
plt.yscale('log')  # Log scale for y-axis
plt.title('Test Error vs. N for Different Sample Sizes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_error_vs_N_logscale.png")
plt.show()

# Save to CSV
flat_results = [(N, s, e) for s, err_list in results.items() for N, e in zip(N_values, err_list)]
df = pd.DataFrame(flat_results, columns=["N", "samples", "test_error[%]"])
df.to_csv("test_error_results1.csv", index=False)
print("\n Results saved to 'test_error_results.csv' and plot saved to 'test_error_vs_N_logscale.png'")

