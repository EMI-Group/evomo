import json
import os
import numpy as np
import jax.numpy as jnp



def check_json_files(directory):
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Check if required keys exist
                keys = ['pop_scale_list', 'pop_scale', 'dim_scale_list', 'dim_scale']
                if not all(key in data for key in keys):
                    results.append(f"{filename}: Missing required keys")
                    continue
                
                # Extract arrays
                pop_scale_list = np.array(data['pop_scale_list'])
                pop_scale = np.array(data['pop_scale'])
                dim_scale_list = np.array(data['dim_scale_list'])
                dim_scale = np.array(data['dim_scale'])
                
                # Check lengths
                if len(pop_scale_list) != len(pop_scale):
                    results.append(f"{filename}: 'pop_scale_list' and 'pop_scale' lengths do not match")
                if len(dim_scale_list) == 6:
                    results.append(f"{filename}: kknd")
                if not (len(dim_scale_list) == (len(dim_scale) + 1) or len(dim_scale_list) == len(dim_scale)):
                    results.append(f"{filename}: 'dim_scale_list' and 'dim_scale' lengths do not match")
                
                # Check for NaN values
                if np.isnan(pop_scale).any():
                    results.append(f"{filename}: 'pop_scale' contains NaN values")
                if np.isnan(dim_scale).any():
                    results.append(f"{filename}: 'dim_scale' contains NaN values")
    
    return results

# Set the directory path where the JSON files are located
directory_path = '../data/acc_performance'
validation_results = check_json_files(directory_path)

# Print the results
for result in validation_results:
    print(result)