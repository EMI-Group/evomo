import json
import os
import numpy as np
import jax.numpy as jnp



def wash_json_files(directory):
#     results = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                his_data = data["history_data"]
                new_data = []
                new_his = {"raw_obj":his_data[0]["raw_obj"], "pf_solutions":his_data[0]["pf_solutions"], "pf_fitness":his_data[0]["pf_fitness"]}
                new_data.append(new_his)
                new_d = {"history_data": new_data, "igd": data["igd"], "time": data["time"]}
            with open(file_path, 'w') as file:
                json.dump(new_d, file)
                
    
#     return results

# Set the directory path where the JSON files are located
directory_path = '../data/effi_scal'
wash_json_files(directory_path)

# file_path = "../data/effi_scal/HypE_DTLZ1_exp0.json"
# with open(file_path, 'r') as file:
#     data = json.load(file)
#     his_data = data["history_data"]
#     new_data = []
#     new_his = {"raw_obj":his_data[0]["raw_obj"], "pf_solutions":his_data[0]["pf_solutions"], "pf_fitness":his_data[0]["pf_fitness"]}
#     new_data.append(new_his)
#     new_d = {"history_data": new_data, "igd": data["igd"], "time": data["time"]}
    
# with open(file_path, 'w') as file:
#     json.dump(new_d, file)