import os
import pandas as pd 
import numpy as np

import functions
from getGndSequences import getGndSequences
from getGndStructures import getGndStructures
from getCorrectedEnergyMap import getCorrectedEnergyMap
from matplotlib import pyplot as plt 

def MAIN(EXPERIMENT_IDXS):

    full_data = pd.read_json('experiment.json')
    DICT_SIZE = full_data['DICT_SIZE'][EXPERIMENT_IDXS[0]]
    MAX_CYCLES = full_data['MAX_CYCLES'][EXPERIMENT_IDXS[0]]
    
    eMap = functions.get_eMap(DICT_SIZE) 

    for EXPERIMENT_IDX in EXPERIMENT_IDXS:

        experiments = pd.read_json('experiment.json')
        EXPERIMENT_NAME = experiments['NAME'][EXPERIMENT_IDX]
        
        if DICT_SIZE != experiments['DICT_SIZE'][EXPERIMENT_IDX]:
            raise Exception('Fatal error, wrong dictionary size')

        if not os.path.isdir(EXPERIMENT_NAME):
            os.mkdir(EXPERIMENT_NAME)
            os.mkdir(os.path.join(EXPERIMENT_NAME,'cycle_0'))
                            
        filePath = os.path.join(EXPERIMENT_NAME,'cycle_0', f'dict_size_{DICT_SIZE}.txt')
        np.savetxt(filePath,eMap,fmt= '%.5f',delimiter=' ')

    fig = plt.figure()
    for iteration in range(MAX_CYCLES):
        for EXPERIMENT_IDX in EXPERIMENT_IDXS:
            getGndSequences(EXPERIMENT_IDX,iteration)
            getGndStructures(EXPERIMENT_IDX,iteration)
        getCorrectedEnergyMap(EXPERIMENT_IDXS,iteration)
