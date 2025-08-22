import fileinput
import sys

# Fix the percentage in detrimental_params.py
with open('detrimental_params.py', 'r') as f:
    content = f.read()

# Change from 10% to 1%
content = content.replace(
    'percentage = self.config.detrimental_threshold_percentile / 100.0',
    'percentage = 0.01  # Use 1% instead of 10%'
)

with open('detrimental_params.py', 'w') as f:
    f.write(content)

print("Updated detrimental_params.py to use 1% masking instead of 10%")
