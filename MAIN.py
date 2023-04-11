import os
import pandas as pd 
import numpy as np
import json

import functions
from getGndSequences import getGndSequences
from getGndStructures import getGndStructures
from getCorrectedEnergyMap import getCorrectedEnergyMap
from matplotlib import pyplot as plt 


def MAIN(EXPERIMENT_IDX,heatmap,run):
    working_dir = os.path.join(f'heatmap_{heatmap}',f'run_{run}')
    filePath = os.path.join(working_dir, 'experiment.json')
  
    with open(filePath, 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]
        
    MAX_CYCLES  = data['MAX_CYCLES']
    
    fig = plt.figure()
    
    for iteration in range(MAX_CYCLES):
        getGndSequences(EXPERIMENT_IDX,heatmap,run,iteration)
        getGndStructures(EXPERIMENT_IDX,heatmap,run,iteration)
        getCorrectedEnergyMap(EXPERIMENT_IDX,heatmap,run,iteration)
        functions.get_drawing(run,iteration)
