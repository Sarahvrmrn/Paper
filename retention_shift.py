# import modules
import os               # for interaction with operating system, change/select directories, etc. https://hellocoding.de/blog/coding-language/python/os-module-richtig-sicher-nutzen
from os.path import join
import numpy as np      # for numerical operations
import pandas as pd     # for data cleaning and analysis
import seaborn as sns   # part of numpy, for distributions
from datetime import datetime
import matplotlib.pyplot as plt


#import custom functions
from helpers import Helpers as hp  # supporting file with additional, reoccuring functions

def shift_retentiontime(input_dir, shift_value, output_csv_dir):
    # Ensure the output directory exists
    os.makedirs(output_csv_dir, exist_ok=True)

    # Iterate over all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(input_dir, filename)
            
            # Read the CSV file
            df = pd.read_csv(input_csv_path)
    
        # Check if the first column contains numeric values from the first row onwards
        if pd.api.types.is_numeric_dtype(df.iloc[0:, 0]):
            # Add the shift_value to every value in the first column starting from the first row
            df.iloc[0:, 0] += shift_value
        else:
            print("The first column does not contain numeric values from the first row onwards.")
            continue
    
        # Get the original file name without the extension
        original_file_name = os.path.splitext(filename)[0]

        # Determine if the add_value is positive or negative and set the descriptor
        descriptor = "plus" if shift_value >= 0 else "minus"
    
        # Create the absolute value for the output file name
        abs_shift_value = abs(shift_value)

        # Generate the output file name by appending the shift_value to the original file name
        output_csv_name = f'{original_file_name}_{descriptor}_{abs_shift_value}.csv'
        output_csv_path = os.path.join(output_csv_dir, output_csv_name)
    
        # Save the modified dataframe to a new CSV file
        df.to_csv(output_csv_path, index=False)
        print(f"Modified CSV saved to {output_csv_path}")

# select input and output directories
input_dir = 'C:\\Users\\sverme-adm\\Desktop\\Knolle'
output_csv_dir = 'C:\\Users\\sverme-adm\\Desktop\\Data_Sorten\\Knolle'

shift_value = -300      # shift the retention time (first column) by this value 

# execute function
shift_retentiontime(input_dir, shift_value, output_csv_dir)