import json
import matplotlib.pyplot as plt
import numpy as np
import math

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

# Load the JSON file
with open("1129_191410.json", 'r') as file:
  data = json.load(file)

# Initialize lists to store extracted values
steps = []
values = []

# If each item in the list is a pair like [step, value]
for obj in data:
    steps.append(obj[1])  # Assuming step is at index 0
    values.append(obj[2])  # Assuming value is at index 1


# Define the smoothing factor (weight/alpha)
weight = 0.9  # You can adjust this for different smoothing effects (e.g., 0.9, 0.95)

# Apply the smoothing function
smoothed_values = smooth(values, weight)

# Plot the raw data and smoothed data
plt.figure(figsize=(10, 6))

# Plot raw data
plt.plot(steps, values, label='Acc Value', color='#97BDD7', alpha=0.5)

# Plot smoothed data using the custom EMA function
plt.plot(steps, smoothed_values, label=f'Smoothed Acc Value (weight={weight})', color='#00629b')

plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Step vs Acc Value with Exponential Moving Average (EMA) Smoothing')
plt.legend()

plt.grid(True)
plt.show()
