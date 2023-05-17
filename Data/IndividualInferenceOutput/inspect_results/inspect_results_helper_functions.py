import json
import numpy as np
import os
import os.path
from os.path import exists
from scipy.stats import gaussian_kde
import datetime

## functions used in the inspect_results notebook 

## TODO: add documentation for functions

def find_all_missing(pop, events, fname_template='../{1}/job_{0:05d}_result.json', returnExceptions=True):
    
    missing = []
    not_missing = []
    
    for job in events: 
        JOB=int(job) 

        f = fname_template.format(JOB, pop)
        file_exists = exists(f)
        if not file_exists:
            missing += [JOB]
            #print(JOB)
        else:
            try:
                # Read file
                with open(f,'r') as jf:
                    result = json.load(jf)
                chiEff = np.array(result['posterior']['content']['chi_eff'])
                not_missing += [JOB]

            except Exception as e:
                print(JOB)
                if returnExceptions:
                    missing += [JOB]
                print(e)
            
    return missing, not_missing


def reflective_kde_on_grid(points, grid):
    
    A = grid[0]
    B = grid[-1]
    
    kde_on_grid = gaussian_kde(points)(grid) + gaussian_kde(-points + 2*A)(grid) + gaussian_kde(-points + 2*B)(grid)
    
    return kde_on_grid


def get_snr(H1, L1, V1): 
    
    H1_snr = np.sqrt(H1['real']**2 + H1['imag']**2)
    L1_snr = np.sqrt(H1['real']**2 + H1['imag']**2)
    V1_snr = np.sqrt(H1['real']**2 + H1['imag']**2)
    
    network_SNR = np.sqrt(H1_snr**2 + L1_snr**2 + V1_snr**2)
    
    return network_SNR



def is_file_older_than_one_day(file_path):
    # Get the timestamp of the file
    timestamp = os.path.getmtime(file_path)

    # Convert the timestamp to a datetime object
    file_time = datetime.datetime.fromtimestamp(timestamp)

    # Get the current time
    current_time = datetime.datetime.now()

    # Calculate the time difference between the current time and the file's timestamp
    time_difference = current_time - file_time

    # Check if the time difference is greater than one day
    if time_difference.days > 0:
        return True
    else:
        return False


