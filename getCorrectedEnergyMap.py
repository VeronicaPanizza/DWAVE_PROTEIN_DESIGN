import numpy as np
import os
import json
import pandas as pd
import math
import random
import functions
import matplotlib.pyplot as plt


#   INPUTS:  
#       EXPERIMENT_IDX:     identifies the experiment we are carring on (see experiment.json file).
#                           This index is used to extract all information required (target structure, 
#                           size of box, dictionary size,...);
#       CYCLE:              The whole algorithm is supposed to repeat "sequences optimization",  
#                           "structure optimization" and "correction of energy map" until convergence. 
#                           Since intermediate results are stored in appropriate folders, it is possible 
#                           to access results of "sequences optimization" (i.e. input sequences);

#   OUTPUTS:
#       NEW_ENERGY_MAP:     corrected energy map;           

def getCorrectedEnergyMap(EXPERIMENT_IDXS, CYCLE):
    
    with open('experiment.json', 'r') as f:      
        full_data = json.load(f)
    
    TMP_DICT_SIZE = full_data[EXPERIMENT_IDXS[0]]['DICT_SIZE']

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,'CORRECTING ENERGY MAP',sep)
        

    x = np.zeros((0,TMP_DICT_SIZE ** 2))         
    y = np.zeros((0,TMP_DICT_SIZE ** 2))                                                                  

    for EXPERIMENT_IDX in EXPERIMENT_IDXS:
        
        # --------------------------------------------------------------------------------------#
        # LOAD CURRENT EXPERIMENT SETTINGS;

        
        data = full_data[EXPERIMENT_IDX]
        
        EXPERIMENT_NAME = data["NAME"]                      # Experiment name: will be the output folder;
        N_X             = int(data["N_X"])                  # Width of the lattice containing the Hamiltonian path;
        N_Y             = int(data["N_Y"])                  # Height of the lattice containing the Hamiltonian path;
        S_IND           = int(data["TARGET_STRUCTURE"])     # Index associated with the DESIGNABLE target structure;
        DICT_SIZE       = int(data["DICT_SIZE"])            # Dictionary size;
        
        SA_ON           = bool(data["SA_ON"])               # True: we extract input sequences from "sa.json" file;
        QA_ON           = bool(data["QA_ON"])               # True: we extract input sequences from "qa.json" file;
        HQA_ON          = bool(data["HQA_ON"])              # True: we extract input sequences from "hqa.json" file;
        
        TEMPERATURE     = 0.15323
        ALPHA           = 1 

        eMapReal = np.loadtxt(os.path.join('ENERGY_MAP',f'dict_size_{DICT_SIZE}.txt'))                  # Load real energy map;
        eMapReal = eMapReal.reshape(1,-1)
        ENERGY_GAP_REAL = TEMPERATURE * (np.max(eMapReal) - np.min(eMapReal)) * (N_X - 1) * (N_Y -1)    # Evaluate real energy gap;                                        
        
        working_dir = os.path.join(EXPERIMENT_NAME, f'cycle_{CYCLE}')                                 

        eMap = np.loadtxt(os.path.join(working_dir,f'dict_size_{DICT_SIZE}.txt'))                       # Load current energy map;
        eMap = np.reshape(eMap,(1,-1))   
        ENERGY_GAP = TEMPERATURE * (np.max(eMap) - np.min(eMap)) * (N_X - 1) * (N_Y -1)                 # Evaluate current energy gap;                                        
        
        target_cmap = np.loadtxt(os.path.join('DATA',f'X_{N_X}_Y_{N_Y}',f'contact_map_{S_IND}.txt'))    # Load target contact map;
        target_cmap = np.reshape(target_cmap,(1, N_X * N_Y, N_X * N_Y))

        
        # --------------------------------------------------------------------------------------#
        # LOAD CURRENT GROUND-STATE ENERGY;
        
        if SA_ON:
            fileName = 'sa.json'
        elif QA_ON:
            fileName = 'qa.json'
        elif HQA_ON:
            fileName = 'hqa.json'
        else:
            raise Exception('No file has been selected') 
        
        # --------------------------------------------------------------------------------------#
        # LOAD DATA CONCERNING GROUND-STATES;
        start = max(CYCLE - 3,0)
        for cycle in range(start,CYCLE+1):
            filePath = os.path.join(EXPERIMENT_NAME, f'cycle_{cycle}', 'gnd_structures_from_' + fileName)
            if cycle == start:
                gndDataFrame = pd.read_json(filePath)
            else:
                gndDataFrame = pd.concat([gndDataFrame,pd.read_json(filePath)],ignore_index=True)
        
        gndDataFrame = gndDataFrame.sort_values('energy',ignore_index=True)

        
        sequences = np.array(list(gndDataFrame['sequence']))                                                # Exctract all sequences;
        sequences_reduced, idx_reverse = np.unique(sequences,axis=0,return_inverse=True)                    # Remove redundant sequences and keep track of the original set of sequences;
        N_SEQUENCES = len(sequences_reduced)
        
        n_vec_target = functions.n_vector(sequences_reduced, target_cmap, DICT_SIZE)                        # Mount all sequences obtained from Q.A. on the target structure;
        n_vec_target = n_vec_target[0,:,:]                                                                  # Add a new axis to n_vec_target;

        for seq_idx in range(N_SEQUENCES):
        
            # Isolate data associated with 'seq';
            df_idx = np.where(idx_reverse == seq_idx)[0]                                                    # Select a sub-dataframe where all data have the same sequence;

            reducedGndDataFrame = gndDataFrame[gndDataFrame.index.isin(df_idx)]                              
            reducedGndDataFrame = reducedGndDataFrame.sort_values('energy',ignore_index=True)
            N_STRUCTURES = len(reducedGndDataFrame)

            energy_threshold_1 = min(reducedGndDataFrame['energy']) + ENERGY_GAP_REAL
            energy_threshold_2 = np.sum(eMapReal * n_vec_target[seq_idx],axis=1)[0]
            energy_threshold = min(energy_threshold_1,energy_threshold_2)
            
            # energy_threshold = np.sum(eMapReal * n_vec_target[seq_idx],axis=1)[0]
            
            r_reducedGndDataFrame = reducedGndDataFrame[reducedGndDataFrame['energy'] < energy_threshold]   # Select data associated with structures that
                                                                                                            # compete with the target one;
            if len(r_reducedGndDataFrame) != 0:
                    
                n_vecs = np.array(r_reducedGndDataFrame['n_vec'].to_list())     
                n_vecs = np.unique(n_vecs,axis = 0)       
                x = np.concatenate((x,n_vec_target[seq_idx] - n_vecs)) 

            energy_threshold = min(reducedGndDataFrame['energy']) + ENERGY_GAP_REAL
            r_reducedGndDataFrame = reducedGndDataFrame[reducedGndDataFrame['energy'] < energy_threshold]
            
            if len(r_reducedGndDataFrame) == 1:  
                # print('Sequence', r_reducedGndDataFrame['sequence'][0],f'folds uniquely in', r_reducedGndDataFrame['structure'][0])

                n = r_reducedGndDataFrame['n_vec'][0]

                for struct_idx in range(N_STRUCTURES):                                                      # Consider all structures different from the native one;
                    if struct_idx == r_reducedGndDataFrame['structure'][0]:
                        continue
                    nt = reducedGndDataFrame[reducedGndDataFrame['structure']==struct_idx].reset_index(drop = True)
                    nt = nt['n_vec'][0]
                    nt, n = np.array(nt),np.array(n)
                    newLine = (nt - n).reshape((1,-1))                                                       
                    y = np.concatenate((y, ALPHA * newLine))

    for a in y:                                                                                             # This check can be removed once it is clearly satisfied;
        if (a == np.zeros_like(y)).all():
            print('An error has occured, there are null vectors in y')
            break                

    if len(x) == 0 and len(y)==0:                                                                           # If there are no vectors in x, then the target structure
        print('\n Criteria are satisfied before optimization')                                              # already occupies the lowest position in energy spectrum 
        return                                                                                              # for every sequence.
        
    # --------------------------------------------------------------------------------------#
    # EVALUATE CURRENT ENERGY OF LOW-ENERGY STRUCTURES;

    dist = []

    lr0 = 1 / (2*((DICT_SIZE + DICT_SIZE**.5) / 2) ** .5)
    lr = lr0 / (1 + CYCLE * 3)

    for count in range(1000):
        if np.sum(eMap) < 0:
            eMap = eMap - 2 * np.sum(eMap) / 9
        dist.append(np.linalg.norm(eMap - eMapReal))

        values_x = np.sum(eMap * x, axis = 1)                                                                           # Evaluate the energy of selected sequences on selected 
                                                                                                                        # structures using current energy map;
        values_y = np.sum(eMap*y,axis=1) - ALPHA * ENERGY_GAP_REAL                                          
        
        lowest_value_x, lowest_value_y = 0,0

        if len(values_x) > 0:
            lowest_value_x = min(values_x)
        if len(values_y) > 0:
            lowest_value_y = min(values_y)

        opt_x, opt_y = True, True
        
        if lowest_value_x < lowest_value_y:
            opt_y = False
            lowest_value = lowest_value_x
        else:
            opt_x = False
            lowest_value = lowest_value_y

        if lowest_value >= 0:                                                                                          # All constraints are satisfied;
            print('\n All criteria are satisfied!')
            break

        if opt_x:
            pos_lowest_value_x = [x for x in range(len(values_x)) if values_x[x] == lowest_value_x]                     # Identify the most violated constraint;
            # print(f'{"First process:":15} {lowest_value_x:.3f}\t pos: {pos_lowest_value_x}')
            update = np.mean(x[pos_lowest_value_x],axis=0)

        elif opt_y:
            pos_lowest_value_y = [y for y in range(len(values_y)) if values_y[y] == lowest_value_y]                     # Identify the most violated constraint;
            # print(f'{"Second process:":15} {lowest_value_y:.3f}\t pos: {pos_lowest_value_y}')
            update = np.mean(y[pos_lowest_value_y],axis=0)
            
        update = update / np.linalg.norm(update)
        eMap = eMap + update * lr                                                                                       # Update current energy map using `x` of most violated constraint;          
        eMap = eMap / np.linalg.norm(eMap)                                                                              # Normalize new energy map;
        eMap = np.reshape(eMap,(1,-1))                                                                                  # Reshape new energy map (array-like);


    print('\n Number of iterations:',(count+1))


    # --------------------------------------------------------------------------------------#
    # SAVE ENERGY MAP IN NEXT DIRECTORY;

    eMap = np.reshape(eMap,(DICT_SIZE,DICT_SIZE))

    for EXPERIMENT_IDX in EXPERIMENT_IDXS:
        
        with open('experiment.json', 'r') as f:      
            full_data = json.load(f)
        data = full_data[EXPERIMENT_IDX]
        
        EXPERIMENT_NAME = data["NAME"]                      # Experiment name: will be the output folder;
    
        next_dir = os.path.join(EXPERIMENT_NAME,f'cycle_{CYCLE+1}')

        if not os.path.isdir(next_dir):
            os.mkdir(next_dir)

        file_path = os.path.join(next_dir,f'dict_size_{DICT_SIZE}.txt')
        np.savetxt(file_path,eMap,fmt='%.5f',delimiter = ' ')
    
    plt.plot(dist)
    plt.grid()
    plt.savefig('dist.png')

    dist_mean = np.mean(dist)                                                       # Save additional information about the Froebenius distance between
    std_mean = np.std(dist)                                                         # real energy map and energy map across n-th cycle;
    data_add = {}
    data_add['cycle'] = [CYCLE]
    data_add['mean_dist'] = [dist_mean]
    data_add['std_dist'] = [std_mean]
    data_add = pd.DataFrame.from_dict(data_add)
    
    if not os.path.isfile('distAnalysis.json'):
        data = data_add
    else:
        data = pd.read_json('distAnalysis.json')
        data = pd.concat((data,data_add),ignore_index=True)
    
    data.to_json('distAnalysis.json',indent=2)