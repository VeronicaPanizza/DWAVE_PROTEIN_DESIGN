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
    
    EXPERIMENT_NAME = data["NAME"]                      # Experiment name: will be the output folder;
    N_X             = int(data["N_X"])                  # Width of the lattice containing the Hamiltonian path;
    N_Y             = int(data["N_Y"])                  # Height of the lattice containing the Hamiltonian path;
    COMPOSITION     = data["COMPOSITION"]               # Composition constraint associated with the sequence;
    S_IND           = int(data['TARGET_STRUCTURE'])     # Target structure (necessary to extract the N-simple paths);         
    DICT_SIZE       = int(data["DICT_SIZE"])            # Dictionary size;
    SA_ON           = bool(data["SA_ON"])               # True: we extract input sequences from "sa.json" file;
    QA_ON           = bool(data["QA_ON"])               # True: we extract input sequences from "qa.json" file;
    HQA_ON          = bool(data["HQA_ON"])              # True: we extract input sequences from "hqa.json" file;
    LENGTH          = N_X * N_Y
    TEMPERATURE     = 0.15323                           # Modify: add field in experiment.json;
    
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
    ENERGY_GAP = TEMPERATURE * (N_X - 1) * (N_Y - 1) * (np.max(eMapCurr) - np.min(eMapCurr))
    
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
    gnd_indexes = sequences[sequences['energy'] <= gnd_energy + ENERGY_GAP].index               # Exctract sequences returnd by Q.A. that have energy between gnd_energy 
    gnd_samples = sequences['sample'][gnd_indexes]                                              # and gnd_energy + ENERGY_GAP (evaluated from current energy map);    

    partial_gnd_samples = list()
    compositions = list()

    for sample in gnd_samples:
        partial_gnd_sample = - np.ones(LENGTH)
        composition = np.zeros(DICT_SIZE)

        for s in range(LENGTH):
            for m in range(DICT_SIZE):
                entry_name = f'q_{s}_{m}'
                if entry_name in sample:
                    if sample[entry_name] == 1:
                        partial_gnd_sample[s] = m
                        composition[m] += 1

        partial_gnd_samples.append(partial_gnd_sample)
        compositions.append(composition)
    del gnd_energy, gnd_samples, sequences                              # Remove from memory unnecessary variables;

    partial_gnd_samples = np.array(partial_gnd_samples)                 # Convert lists into np.arrays;
    compositions = np.array(compositions)
   
    # --------------------------------------------------------------------------------------#
    # COMPLETE GROUND-STATE SEQUENCES; 

    # --------------------------------------------------------------------------------------#
    # STEP 0:
    #   Suppose the current sequence currently has composition (i.e. [#A, #B, #C]) [1, 1, 1] and
    #   that the target composition is [5, 4, 4]. By doing the difference [3, 1, 2] - [1, 1, 1] 
    #   = [2, 0, 1], we find that we need 2 monomers A, 0 monomers B, and 1 monomer C. 

    # STEP 1:
    #   We build an array (i.e. 'missing monomers array')'A A C'.

    # STEP 2:
    # Find all non-equivalent permutations of the 'missing monomers array' (i.e. 'A A C'):
    # 'A A C', 'A C A', and 'C A A'.
    
    # STEP 3:
    #   Then, if the partial sequence was A -1 -1 C -1 B (-1 stands for an ignote monomer), we 
    #   will have the following completements:
    #       1) A A A C C B,
    #       2) A A C C A B,
    #       3) A C A C A B.
    # --------------------------------------------------------------------------------------#
    
    # STEP 0;
    missing_compositions = (COMPOSITION - compositions).astype(int)     
    missing_monomers = list()                                           # List of 'Missing monomers' arrays (one for each sequence);

    # STEP 1;
    for mc in missing_compositions:                                      
        mm = - np.ones((sum(mc)))                                       # Allocate 'missing monomers' array;
        idx = 0                                                         
        for flavour,number in enumerate(mc):                            # Initialize 'missing monomers' array;
            if number > 0:                                              
                mm[idx : idx + number] = flavour * np.ones(number)
            idx += number    
        missing_monomers.append(mm)

    # STEP 2;
    comb_missing_monomers = {} 

    for idx, mm in enumerate(missing_monomers):
        name = f'sequence_{idx}'

        comb_missing_monomers[name] = list()
        all_permutations = permutations(mm,len(mm)) 
        
        for perm_idx, perm in enumerate(set(all_permutations)):    
            comb_missing_monomers[name].append(perm) 

    # STEP 3;
    complete_gnd_samples = {}                                           
    for sample_idx, sample in enumerate(partial_gnd_samples):
        name = f'sequence_{sample_idx}'
        complete_gnd_samples[name] = np.broadcast_to(sample,(len(comb_missing_monomers[name]),LENGTH)).copy()
        for perm_idx, perm in enumerate(comb_missing_monomers[name]):
            missing_monomer_idx = 0
            for entry_idx,entry in enumerate(sample):
                if entry == -1:
                    complete_gnd_samples[name][perm_idx, entry_idx] = comb_missing_monomers[name][perm_idx][missing_monomer_idx]
                    missing_monomer_idx += 1

    sequences_list = list()
    for s in complete_gnd_samples:
        sequences_list.append(complete_gnd_samples[s])
    sequences = np.concatenate(sequences_list,axis = 0)
    
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
    dir_content_filtered = [dir_content[x] for x in range(len(dir_content)) if 'contact_map' in dir_content[x]]
    
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




