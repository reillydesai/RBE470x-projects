import subprocess

# Number of times to run the file
num_runs = 100

for i in range(num_runs):
    print(f"Run iteration {i+1}")
    # Replace 'python3' with 'python' if necessary for your environment
    subprocess.run(["python3", "variant3.py"])
