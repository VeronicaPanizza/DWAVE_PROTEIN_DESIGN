import numpy as np
import random as rnd
import pandas as pd
import os
import json
import math
from itertools import permutations, product
import functions


def getGndStructures(EXPERIMENT_IDX,HEATMAP,RUN,CYCLE):
    
    #   INPUTS:
    #       EXPERIMENT_IDX: identifies the experiment we are carring on (see experiment.json file).
    #                       This index is used to extract all information required (target structure, 
    #                       size of box, dictionary size,...);
    #       CYCLE:          The whole algorithm is supposed to repeat "sequence optimization" and 
    #                       "structure optimization" until convergence. Since intermediate results are 
    #                       stored in appropriate folders, it is possible to access results of 
    #                       "sequence optimization" (i.e. input sequences);

    #   OUTPUTS:            for each input sequence, a set of groundstate structures with lowest energy
    #                       is given as output;


    # --------------------------------------------------------------------------------------#
    # LOAD CURRENT EXPERIMENT SETTINGS;
    
    working_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}')
    filePath = os.path.join(working_dir, 'experiment.json')
    with open(filePath, 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]
    
    
    EXPERIMENT_NAME         = data["NAME"]                                      # Experiment name: will be the output folder;
    N_X                     = int(data["N_X"])                                  # Width of the lattice containing the Hamiltonian path;
    N_Y                     = int(data["N_Y"])                                  # Height of the lattice containing the Hamiltonian path;
    S_IND                   = int(data['TARGET_STRUCTURE'])                     # Target structure (necessary to extract the N-simple paths);         
    DICT_SIZE               = int(data["DICT_SIZE"])                            # Dictionary size;
    PROBABILITY_THRESHOLD   = data['PROBABILITY_THRESHOLD']
    
    LENGTH                  = N_X * N_Y

    
    working_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}',f'cycle_{CYCLE}')

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,f'GET GROUND-STATE STRUCTURES FOR EXPERIMENT {EXPERIMENT_IDX}: \t {EXPERIMENT_NAME}',sep)

    # Load real energy map;

    eMap = np.loadtxt(f'ENERGY_MAP/dict_size_{DICT_SIZE}.txt')
    print(f'\n Cycle:\t {CYCLE} \n\n Real energy map:\n{eMap}\n')
    eMap = np.reshape(eMap,(1,1,DICT_SIZE**2))

    # Load current energy map;
    eMapCurr = np.loadtxt(os.path.join(working_dir,f'dict_size_{DICT_SIZE}.txt'))
    eMapCurr = np.reshape(eMapCurr,(DICT_SIZE**2))
 
    
    # --------------------------------------------------------------------------------------#
    # LOAD ENUMERATED STRUCTURES;
    
    folder = os.path.join('Structures/',f'X_{N_X}_Y_{N_Y}/')
    dir_content = os.listdir(folder)
    dir_content_filtered = [dir_content[x] for x in range(len(dir_content)) if 'contact_map' in dir_content[x] and 'avg' not in dir_content[x]]
    
    N_STRUCTURES = len(dir_content_filtered)
    
    maps = np.zeros((N_STRUCTURES, LENGTH, LENGTH))
    
    for file_idx in range(len(dir_content_filtered)):
        file_name = f'contact_map_{file_idx}.txt'
        filePath = os.path.join(folder,file_name)
        maps[file_idx,:,:] = np.loadtxt(filePath, delimiter = ' ')
            
     # --------------------------------------------------------------------------------------#
     # LOAD SEQUENCES;
     
    file_path = os.path.join(working_dir,'sa.json')
    data = pd.read_json(file_path)
    energy = np.array(data['energy'])
    
    thrEnergy = min(energy)
    data = data[data['energy'] == thrEnergy]
    
    del energy
    
    sequences = np.array(data['sequence'].tolist()).astype(int)
    
    # --------------------------------------------------------------------------------------#
    # EXTENSIVE EVALUATION OF ENERGY FUNCTIONAL;
    
    gnd_data                = {}                                                    # Build output as a dictionary;    
    gnd_data['energy']      = []
    gnd_data['n_vec']       = []
    gnd_data['sequence']    = []
    gnd_data['structure']   = []

    des_data                = {}                                                    # Build output as a dictionary:
    des_data['sequence']    = []                                                    #   This will contain complete information about
    des_data['n_vec']       = []                                                    #   the outcome of the 'designing' procedure (complete 
    des_data['energy']      = []                                                    #   sequence, n_vector on target structure and its energy);
    
    n_vec_direct    = functions.n_vector(sequences,maps,DICT_SIZE)                  # Direct housing;
    # n_vec_inverse   = functions.n_vector(sequences[:,::-1],maps,DICT_SIZE)          # Inverse housing; 

    energy_direct   = np.sum(n_vec_direct * eMap, axis = 2)                         # Evaluate energy for direct housing of sequences;
    # energy_inverse  = np.sum(n_vec_inverse * eMap,axis = 2)                         # Evaluate energy for inverse housing of sequences;


    for sequence_idx, sequence in enumerate(sequences):
        for structure_idx in range(N_STRUCTURES):
            eDirect  = energy_direct[structure_idx,sequence_idx] 
            # eInverse = energy_inverse[structure_idx,sequence_idx]

            gnd_data['energy'].append(eDirect)
            gnd_data['n_vec'].append(n_vec_direct[structure_idx,sequence_idx])
            gnd_data['sequence'].append(sequence)
            gnd_data['structure'].append(structure_idx)
        
            # gnd_data['energy'].append(eInverse)
            # gnd_data['n_vec'].append(n_vec_inverse[structure_idx,sequence_idx])
            # gnd_data['sequence'].append(sequence[::-1])                               
            # gnd_data['structure'].append(structure_idx)

            target_n_vec = n_vec_direct[S_IND,sequence_idx]
            target_energy = np.sum(eMapCurr * target_n_vec)

            des_data['sequence'].append(sequence)
            des_data['n_vec'].append(target_n_vec)
            des_data['energy'].append(target_energy)
            
    gnd_dataframe = pd.DataFrame.from_dict(gnd_data)
    gnd_dataframe = gnd_dataframe.sort_values('energy',ignore_index = True)
    outputFilePath = os.path.join(working_dir,'gnd_structures_from_sa.json')
    gnd_dataframe.to_json(outputFilePath, indent = 2)





