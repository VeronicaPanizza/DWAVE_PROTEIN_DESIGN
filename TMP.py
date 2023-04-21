import os
from MAIN import MAIN
from getCorrectedEnergyMap import getCorrectedEnergyMap
import functions
import numpy as np
import pandas as pd
import shutil

use_real_pot    = False
use_seed        = True

EXPERIMENT_IDX  = 5
N_RUNS          = 1
SEED            = 7
heatmap         = 0

full_data   = pd.read_json('experiment.json')
DICT_SIZE   = full_data['DICT_SIZE'][EXPERIMENT_IDX]
MAX_CYCLES  = full_data['MAX_CYCLES'][EXPERIMENT_IDX]

if os.path.isdir(f'heatmap_{heatmap}'):
    shutil.rmtree(f'heatmap_{heatmap}')
os.mkdir(f'heatmap_{heatmap}')

for run in range(SEED,SEED+N_RUNS):  
    if use_real_pot:
        eMap = np.loadtxt(f'ENERGY_MAP/dict_size_{DICT_SIZE}.txt',delimiter = '\t')
    elif use_seed:
        eMap = np.loadtxt(f'Start_emaps/dict_size_{DICT_SIZE}_{SEED}.txt')        
    else:
        eMap = functions.get_eMap(DICT_SIZE)
            
    working_dir = os.path.join(f'heatmap_{heatmap}',f'run_{run}')
    os.mkdir(working_dir)
    
    filePath = os.path.join(working_dir, 'experiment.json')
    shutil.copyfile('experiment.json', filePath)

    for cycle in range(MAX_CYCLES + 1):
                        
        cycle_dir = os.path.join(working_dir,f'cycle_{cycle}')
        os.mkdir(cycle_dir)
        
        if cycle == 0:
            file_path = os.path.join(working_dir,'cycle_0', f'dict_size_{DICT_SIZE}.txt')
            np.savetxt(file_path,eMap,fmt= '%.5f',delimiter=' ')
        
    MAIN(EXPERIMENT_IDX,heatmap,run)

