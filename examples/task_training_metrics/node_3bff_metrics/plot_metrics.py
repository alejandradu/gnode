import os
import pandas as pd
import matplotlib.pyplot as plt

# Get a list of all CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

for file in csv_files:
    # Load the data
    with open(file, 'r') as f:
        data = pd.read_csv(f)

    # Fill missing values with the previous ones
    data.fillna(method='ffill', inplace=True)

    # Plot validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'], data['valid/loss'], label='Validation Loss')

    # Plot test loss
    if 'test/loss' in data.columns:
        plt.plot(data['epoch'], data['test/loss'], label='Test Loss')

    # Extract parameters from the filename
    parameters = file.split(',')  # Change this if your delimiter is not an underscore
    parameters[-1] = parameters[-1].replace('.csv', '')  # Remove the .csv extension from the last parameter
    title = ' '.join(parameters)

    plt.title(f'Validation and Test Loss Over Epochs for {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()