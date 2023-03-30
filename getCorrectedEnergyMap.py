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

def getCorrectedEnergyMap(EXPERIMENT_IDX,HEATMAP,RUN,CYCLE):

    working_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}')
    filePath = os.path.join(working_dir, 'experiment.json')
    with open(filePath, 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]    
    
    # --------------------------------------------------------------------------------------#
    # LOAD CURRENT EXPERIMENT SETTINGS;

    
    data = full_data[EXPERIMENT_IDX]
    
    EXPERIMENT_NAME = data["NAME"]                                              # Experiment name: will be the output folder;
    N_X                     = int(data["N_X"])                                  # Width of the lattice containing the Hamiltonian path;
    N_Y                     = int(data["N_Y"])                                  # Height of the lattice containing the Hamiltonian path;
    S_IND                   = int(data["TARGET_STRUCTURE"])                     # Index associated with the DESIGNABLE target structure;
    DICT_SIZE               = int(data["DICT_SIZE"])                            # Dictionary size;
    PROBABILITY_THRESHOLD   = data['PROBABILITY_THRESHOLD']
    MODE                    = str(data['MODE'])
    MAX_CYCLES              = int(data['MAX_CYCLES'])
    
    BETA = DICT_SIZE
    
    memory = MAX_CYCLES                                                         # S.A. run are retained throughout the calculation;
    
    # --------------------------------------------------------------------------------------#
    # INITIALIZE VARIABLES AND ARRAYS; 
    
    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,'CORRECTING ENERGY MAP',sep)
    
    x = PROBABILITY_THRESHOLD / (1 - PROBABILITY_THRESHOLD)
    minimum_gap = np.log(x) / BETA
    

    eMapReal = np.loadtxt(os.path.join('ENERGY_MAP',f'dict_size_{DICT_SIZE}.txt'))           # Load real energy map;
    
    eMapReal = eMapReal.reshape(1,-1)
    
    working_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}',f'cycle_{CYCLE}')                                 

    eMap = np.loadtxt(os.path.join(working_dir,f'dict_size_{DICT_SIZE}.txt'))               # Load current energy map;
    eMap = np.reshape(eMap,(1,-1))   
    
    target_cmap = np.loadtxt(os.path.join('Structures',f'X_{N_X}_Y_{N_Y}',f'contact_map_{S_IND}.txt'))          # Load target contact map;
    target_cmap = np.reshape(target_cmap,(1, N_X * N_Y, N_X * N_Y))

    
    # --------------------------------------------------------------------------------------#
    # LOAD DATA CONCERNING GROUND-STATES;
    
    fileName = 'sa.json'
    
    start = max(CYCLE - memory,0)
    
    for cycle in range(start,CYCLE+1):
        filePath = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}',f'cycle_{cycle}','gnd_structures_from_' + fileName)
        if cycle == start:
            gndDataFrame = pd.read_json(filePath)
        else:
            gndDataFrame = pd.concat([gndDataFrame,pd.read_json(filePath)],ignore_index=True)
    
    gndDataFrame = gndDataFrame.sort_values('energy',ignore_index=True)

    
    sequences = np.array(list(gndDataFrame['sequence']))                                        # Exctract all sequences;
    sequences_reduced, idx_reverse = np.unique(sequences,axis=0,return_inverse=True)            # Remove redundant sequences and keep track of the original set of sequences;
    N_SEQUENCES = len(sequences_reduced)
    
    n_vec_target = functions.n_vector(sequences_reduced, target_cmap,DICT_SIZE)                 # Mount all sequences obtained from Q.A. on the target structure;
    n_vec_target = n_vec_target[0,:,:]                                                          # Add a new axis to n_vec_target;
    
    
    # --------------------------------------------------------------------------------------#
    # EXTRACT PARAMETERS ENETERING THE SET OF INEQUALITIES;
    
    
    x = np.empty((0,DICT_SIZE ** 2))         
    y = np.empty((0,DICT_SIZE ** 2))        
    
    
    for seq_idx in range(N_SEQUENCES):
    
        # Isolate data associated with 'seq';
        df_idx = np.where(idx_reverse == seq_idx)[0]                            # Select a sub-dataframe where all data have the same sequence;

        reducedGndDataFrame = gndDataFrame[gndDataFrame.index.isin(df_idx)]                              
        reducedGndDataFrame = reducedGndDataFrame.sort_values('energy',ignore_index=True)
        # N_STRUCTURES = len(reducedGndDataFrame)


        # First learning process: learningy by energy;
        
        # energy_threshold1 = np.sum(eMapReal * n_vec_target[seq_idx],axis=1)[0]        
        # energy_threshold2 = min(reducedGndDataFrame['energy']) + 1
        # energy_threshold = min(energy_threshold1, energy_threshold2)
        
        energy_threshold = np.sum(eMapReal * n_vec_target[seq_idx],axis=1)[0]        
        
        r_reducedGndDataFrame = reducedGndDataFrame[reducedGndDataFrame['energy'] \
                                                    < energy_threshold]         # Select data associated with structures that
                                                                                # compete with the target one;
        
        if len(r_reducedGndDataFrame) != 0:                                     # There are structures that compete with the target one;       
            n_vecs = np.array(r_reducedGndDataFrame['n_vec'].to_list())     
            n_vecs = np.unique(n_vecs,axis = 0)       
            x = np.concatenate((x,n_vec_target[seq_idx] - n_vecs)) 


        # Second learning process: learning by stability;

        gnd_energy       = min(reducedGndDataFrame['energy'])
        energies         = np.array(reducedGndDataFrame['energy'])
        boltzmann_weight = np.exp(-BETA*gnd_energy)/sum(np.exp(-BETA*energies))
        
        if boltzmann_weight > PROBABILITY_THRESHOLD:
            n_vec_gnd   = np.array(reducedGndDataFrame['n_vec'][0])
            
            r_reducedGndDataFrame = reducedGndDataFrame[reducedGndDataFrame.index > 0]
            n_vec       = np.array(r_reducedGndDataFrame['n_vec'].tolist())
            y = np.concatenate((y, n_vec - n_vec_gnd))
            
            

    if len(x) == 0 and len(y)==0:                                               # If there are no vectors in x, then the target structure
        print('\n Criteria are satisfied before optimization')                  # already occupies the lowest position in energy spectrum 
        return                                                                  # for every sequence.
        
    # --------------------------------------------------------------------------------------#
    # EVALUATE CURRENT ENERGY OF LOW-ENERGY STRUCTURES;

    dist = []

    lr0 = 1 / (2*((DICT_SIZE + DICT_SIZE**.5) / 2) ** .5)
    lr = lr0 / (1 + CYCLE * 3)

    for count in range(1000):
        if np.sum(eMap) < 0:
            eMap = eMap - 2 * np.sum(eMap) / 9
        dist.append(np.linalg.norm(eMap - eMapReal))

        values_x = np.sum(eMap * x, axis = 1)                                   # Evaluate the energy of selected sequences on selected 
                                                                                # structures using current energy map;
        values_y = np.sum(eMap*y,axis=1) - minimum_gap                                          
        
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

        if lowest_value >= 0:                                                   # All constraints are satisfied;
            print('\n All criteria are satisfied!')
            break

        if opt_x:
            pos_lowest_value_x = [x for x in range(len(values_x)) \
                                  if values_x[x] == lowest_value_x]             # Identify the most violated constraint;
            update = np.mean(x[pos_lowest_value_x],axis=0)

        elif opt_y:
            pos_lowest_value_y = [y for y in range(len(values_y)) \
                                  if values_y[y] == lowest_value_y]             # Identify the most violated constraint;
            update = np.mean(y[pos_lowest_value_y],axis=0)
            
        update = update / np.linalg.norm(update)
        eMap = eMap + update * lr                                               # Update current energy map using `x` of most violated constraint;          
        eMap = eMap / np.linalg.norm(eMap)                                      # Normalize new energy map;
        eMap = np.reshape(eMap,(1,-1))                                          # Reshape new energy map (array-like);


    print('\n Number of iterations:',(count+1))


    # --------------------------------------------------------------------------------------#
    # SAVE ENERGY MAP IN NEXT DIRECTORY;

    eMap = np.reshape(eMap,(DICT_SIZE,DICT_SIZE))
                                          
    next_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}',f'cycle_{CYCLE+1}')

    file_path = os.path.join(next_dir,f'dict_size_{DICT_SIZE}.txt')
    np.savetxt(file_path,eMap,fmt='%.5f',delimiter = ' ')
    
    plt.plot(dist)
    plt.grid()
    
    file_path = os.path.join(f'heatmap_{HEATMAP}',f'matrix_convergence_run{RUN}.png')
    plt.savefig(file_path)

    dist_mean = np.mean(dist)                                                   # Save additional information about the Froebenius distance between
    std_mean = np.std(dist)                                                     # real energy map and energy map across n-th cycle;
    data_add = {}
    data_add['cycle'] = [CYCLE]
    data_add['mean_dist'] = [dist_mean]
    data_add['std_dist'] = [std_mean]
    data_add = pd.DataFrame.from_dict(data_add)
    
    file_path = os.path.join(f'heatmap_{HEATMAP}',f'distAnalysis_run{RUN}.json')
    if not os.path.isfile(file_path):
        data = data_add
    else:
        data = pd.read_json(file_path)
        data = pd.concat((data,data_add),ignore_index=True)
    
    data.to_json(file_path,indent=2)
