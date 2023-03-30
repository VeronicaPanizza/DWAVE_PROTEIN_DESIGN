import numpy as np
import random as rnd
import pandas as pd
import os
import json
import math
from itertools import permutations, product
import functions

def getGndStructures(EXPERIMENT_IDX,CYCLE):
    
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
    
    with open('experiment.json', 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]
    
    EXPERIMENT_NAME         = data["NAME"]                      # Experiment name: will be the output folder;
    N_X                     = int(data["N_X"])                  # Width of the lattice containing the Hamiltonian path;
    N_Y                     = int(data["N_Y"])                  # Height of the lattice containing the Hamiltonian path;
    S_IND                   = int(data["TARGET_STRUCTURE"])     # Target structure (necessary to extract the N-simple paths);         
    PROBABILITY_THRESHOLD   = data["PROBABILITY_THRESHOLD"]

    DICT_SIZE               = int(data["DICT_SIZE"])            # Dictionary size;
    SA_ON                   = bool(data["SA_ON"])               # True: we extract input sequences from "sa.json" file;
    QA_ON                   = bool(data["QA_ON"])               # True: we extract input sequences from "qa.json" file;
    HQA_ON                  = bool(data["HQA_ON"])              # True: we extract input sequences from "hqa.json" file;
    LENGTH                  = N_X * N_Y
    
    if PROBABILITY_THRESHOLD == 0.8 and DICT_SIZE == 3:
        TEMPERATURE = 0.15323
    elif PROBABILITY_THRESHOLD == 0.8 and DICT_SIZE == 4:
        TEMPERATURE = 0.12513
    elif PROBABILITY_THRESHOLD == 0.9 and DICT_SIZE == 3:
        TEMPERATURE = 0.19
    elif PROBABILITY_THRESHOLD == 0.9 and DICT_SIZE == 4:
        TEMPERATURE = 0.156669679
    else:
        print('Unfitted case, please calibrate the temperature.')
        raise Exception('Unfitted case, please calibrate the temperature')

    working_dir = os.path.join(EXPERIMENT_NAME,f'cycle_{CYCLE}')

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,f'GET GROUND-STATE STRUCTURES FOR EXPERIMENT {EXPERIMENT_IDX}: \t {EXPERIMENT_NAME}',sep)

    # Load real energy map;

    eMap = np.loadtxt(f'ENERGY_MAP/dict_size_{DICT_SIZE}.txt')
    print(f'\n Cycle:\t {CYCLE} \n\n Real energy map:\n{eMap}\n')
    eMap = np.reshape(eMap,(1,1,DICT_SIZE**2))

    # Load current energy map;
    eMapCurr = np.loadtxt(os.path.join(working_dir,f'dict_size_{DICT_SIZE}.txt'))
    eMapCurr = np.reshape(eMapCurr,(DICT_SIZE**2))

    # Evaluate energy gap;
    # ENERGY_GAP = TEMPERATURE * (N_X - 1) * (N_Y - 1) * (np.max(eMapCurr) - np.min(eMapCurr))
    
    # --------------------------------------------------------------------------------------#
    # LOAD PARTIAL GROUND-STATE SEQUENCES;
    
    if SA_ON:
        fileName = 'sa.json'
    elif QA_ON:
        fileName = 'qa.json'
    elif HQA_ON:
        fileName = 'hqa.json'
    else:
        raise Exception('No file has been selected')

    sequences = pd.read_json(os.path.join(working_dir,fileName))
    gnd_energy = min(sequences['energy'])   
    gnd_indexes = sequences[sequences['energy'] == gnd_energy].index                            # Exctract sequences returnd by Q.A. that have energy between gnd_energy 
    gnd_samples = sequences['sample'][gnd_indexes]                                              # and gnd_energy + ENERGY_GAP (evaluated from current energy map);    

    partial_gnd_samples = list()
    compositions = list()

    for sample in gnd_samples:
        sample = list(sample.values())
        partial_gnd_sample = []

        # Adapt to new encoding;
        for s in range(LENGTH):
            color_on = False

            for m in range(DICT_SIZE-1):
                q_pos = s * (DICT_SIZE-1) + m
                q = sample[q_pos]
                if q == 1 and not color_on:
                    color_on = True
                    partial_gnd_sample.append(m)
                elif q==1 and color_on:
                    raise Exception('Excluded volume constraint violated')
            if not color_on:
                partial_gnd_sample.append(DICT_SIZE-1)

        partial_gnd_samples.append(partial_gnd_sample)

    del gnd_energy, gnd_samples, sequences                              # Remove from memory unnecessary variables;

    sequences  = np.array(partial_gnd_samples)                 # Convert lists into np.arrays;
    
    # --------------------------------------------------------------------------------------#
    # REPRODUCE GROUND-STATE SUBSPACES (reflexion symmetry along N-simple paths)
    
    file = os.path.join('DATA',f'X_{N_X}_Y_{N_Y}',f'simple_paths_{S_IND}.txt')
    simple_paths = list()

    with open(file,'r') as fp:
        [simple_paths.append(np.fromstring(x,dtype=int,sep=' ')) for x in fp.readlines()]
    
    nr_simple_paths = len(simple_paths)

    degeneracy = pow(2, nr_simple_paths)
    degeneracy_arrays = product(range(2),repeat = nr_simple_paths) 
    
    N_SEQUENCES = np.shape(sequences)[0]
    new_shape = [degeneracy, N_SEQUENCES, N_X * N_Y]
    sequences = np.broadcast_to(sequences, tuple(new_shape)).copy()

    for arr_idx,arr in enumerate(degeneracy_arrays):
        for path_idx, path in enumerate(simple_paths):
            if arr[path_idx] == 1:
                # Reflexion 
                depth = math.floor(len(path)/2)
                for pos1 in range(depth):
                    pos2 = len(path) - pos1 - 1
                    col1, col2 = path[pos1], path[pos2]
                    sequences[arr_idx,:,[col1,col2]] = sequences[0,:,[col2,col1]]
    
    sequences = np.reshape(sequences,(new_shape[0] * new_shape[1],new_shape[2])) 
    sequences = np.unique(sequences,axis = 0)

    sequences = sequences.tolist()
    for s in sequences:
        s_inv = s[::-1]
        if s != s_inv and s_inv in sequences:
            sequences.remove(s_inv)
    sequences = np.array(sequences)


    N_SEQUENCES = np.shape(sequences)[0]
    
    # --------------------------------------------------------------------------------------#
    # LOAD ENUMERATED STRUCTURES;
    
    folder = os.path.join('DATA/',f'X_{N_X}_Y_{N_Y}/')
    dir_content = os.listdir(folder)
    dir_content_filtered = [dir_content[x] for x in range(len(dir_content)) if 'contact_map' in dir_content[x] and 'avg' not in dir_content[x]]
    
    N_STRUCTURES = len(dir_content_filtered)
    
    maps = np.zeros((N_STRUCTURES, N_X * N_Y, N_X * N_Y))
    for file_idx in range(len(dir_content_filtered)):
            file_name = f'contact_map_{file_idx}.txt'
            filePath = os.path.join(folder,file_name)
            maps[file_idx,:,:] = np.loadtxt(filePath, delimiter = ' ')
            
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
    n_vec_inverse   = functions.n_vector(sequences[:,::-1],maps,DICT_SIZE)          # Inverse housing; 

    energy_direct   = np.sum(n_vec_direct * eMap, axis = 2)                         # Evaluate energy for direct housing of sequences;
    energy_inverse  = np.sum(n_vec_inverse * eMap,axis = 2)                         # Evaluate energy for inverse housing of sequences;

    gndEnergyDirect = np.min(energy_direct)                     
    gndEnergyInverse = np.min(energy_direct)

    gndEnergy = min(gndEnergyDirect, gndEnergyInverse)

    for sequence_idx, sequence in enumerate(sequences):
        for structure_idx in range(N_STRUCTURES):
            eDirect  = energy_direct[structure_idx,sequence_idx] 
            eInverse = energy_inverse[structure_idx,sequence_idx]

            gnd_data['energy'].append(eDirect)
            gnd_data['n_vec'].append(n_vec_direct[structure_idx,sequence_idx])
            gnd_data['sequence'].append(sequence)
            gnd_data['structure'].append(structure_idx)
        
            gnd_data['energy'].append(eInverse)
            gnd_data['n_vec'].append(n_vec_inverse[structure_idx,sequence_idx])
            gnd_data['sequence'].append(sequence[::-1])                               
            gnd_data['structure'].append(structure_idx)

            target_n_vec = n_vec_direct[S_IND,sequence_idx]
            target_energy = np.sum(eMapCurr * target_n_vec)

            des_data['sequence'].append(sequence)
            des_data['n_vec'].append(target_n_vec)
            des_data['energy'].append(target_energy)
            
    gnd_dataframe = pd.DataFrame.from_dict(gnd_data)
    gnd_dataframe = gnd_dataframe.sort_values('energy',ignore_index = True)
    outputFilePath = os.path.join(working_dir,'gnd_structures_from_'+fileName)
    gnd_dataframe.to_json(outputFilePath, indent = 2)

    des_dataframe = pd.DataFrame.from_dict(des_data)
    des_dataframe = des_dataframe.sort_values('energy',ignore_index = True)
    outputFilePath = os.path.join(working_dir,'complete_' + fileName)
    des_dataframe.to_json(outputFilePath,indent = 2)


