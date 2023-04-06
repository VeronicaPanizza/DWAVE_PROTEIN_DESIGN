import os
import numpy as np 
from numpy import linalg
import pandas as pd
import json


N_RUNS          = 1
EXPERIMENT_IDX  = 5

with open('experiment.json', 'r') as f:      
    full_data = json.load(f)
    
data = full_data[EXPERIMENT_IDX]

EXPERIMENT_NAME = data['NAME']
DICT_SIZE       = data['DICT_SIZE']
N_X             = data['N_X']
N_Y             = data['N_Y']
S_IND           = data['TARGET_STRUCTURE']
MAX_CYCLES      = data['MAX_CYCLES']
PTHR            = data['PROBABILITY_THRESHOLD']

BETA = DICT_SIZE


# REMOVE PRE-EXISTENT FILES;
if os.path.isfile('deeperAnalysis.json'):
    os.remove('deeperAnalysis.json')

data_out                                = {}
data_out['Run']                         = []
data_out['Cycle']                       = []
data_out['Foldable_percentage']         = []
data_out['Target_foldable_percentage']  = []
data_out['Distance']                    = []


# LOADING RESULTS OF FOLDING PROCESS;
for RUN in range(N_RUNS):
        
    for CYCLE in range(MAX_CYCLES):
        print(f'\n RUN: {RUN} \t CYCLE:', CYCLE,'\n')
        
        f_curr = os.path.join(f'run_{RUN}',f'cycle_{CYCLE}')
        f_next = os.path.join(f'run_{RUN}', f'cycle_{CYCLE+1}')
        
        fp = os.path.join(f_curr,f'dict_size_{DICT_SIZE}.txt')
        tmp_map = np.loadtxt(fp, delimiter=' ')
        
        fp = os.path.join(f_curr,'gnd_structures_from_sa.json')
        df = pd.read_json(fp)
        df = df.sort_values(['energy'],ignore_index = True)
        
        n_structures = max(df['structure']) + 1 
        n_sequences = len(df) / n_structures
        
        sequences = np.vstack(df['sequence'])                                         
        r_sequences, r_idxs = np.unique(sequences, axis = 0, return_inverse=True)       
                                                                                        
        SEQ                 = []
        BOLTZMANN_WEIGHT    = []                                                                        
        OCCURRENCY          = []
        GND_ENERGY          = []                                                                  
        FIRST_POS           = []
                                
        for seq_idx in range(len(r_sequences)):

            idxs = [x for x in range(len(r_idxs)) if r_idxs[x] == seq_idx]              # Select indeces in database associated with a given sequence;
            r_df = df[df.index.isin(idxs)]                                              # Isolate sub-dataframe s.t. sequence is fixed; 
            
            r_df = r_df.sort_values(['energy'], ignore_index = True)
            
            occurrency = len(r_df) / n_structures
            r_df = r_df.drop_duplicates('structure',ignore_index=True)
            
            energies = np.array(r_df['energy'])
            b_weight = np.exp(-BETA * energies[0]) 
            b_weight = b_weight / sum(np.exp(-BETA * energies))
            
            first_pos = r_df[r_df['structure']==S_IND].index[0]                          

            if b_weight > PTHR:                                                         # If the sequence allows a native state, save associated data;
                SEQ.append(seq_idx)    
                BOLTZMANN_WEIGHT.append(b_weight)
                OCCURRENCY.append(occurrency)
                GND_ENERGY.append(min(r_df['energy']))
                FIRST_POS.append(first_pos)

        data= {}                                                                        # Gather this data in a dictionary;
        data['Sequence_id']      = SEQ
        data['Boltzmann_weight'] = BOLTZMANN_WEIGHT
        data['Occurrency']       = OCCURRENCY
        data['Gnd_energy']       = GND_ENERGY
        data['Target_position']  = FIRST_POS
        
        data = pd.DataFrame(data)
        
        foldable_percentage         = 0
        target_foldable_percentage  = 0        
        
        if len(data)!=0:
            foldable_percentage = sum(data['Occurrency']) / n_sequences * 100
            target_data = data[data['Target_position']==0]
            target_foldable_percentage = sum(target_data['Occurrency']) / sum(data['Occurrency'])*100
    
        eMap_curr = np.loadtxt(os.path.join(f_curr,f'dict_size_{DICT_SIZE}.txt'))
        eMap_next = np.loadtxt(os.path.join(f_next,f'dict_size_{DICT_SIZE}.txt'))
        distance = np.linalg.norm(eMap_curr - eMap_next,2)
    
        data_out['Run'].append(RUN)
        data_out['Cycle'].append(CYCLE)
        data_out['Foldable_percentage'].append(foldable_percentage)
        data_out['Target_foldable_percentage'].append(target_foldable_percentage)
        data_out['Distance'].append(distance)
                        
with open('deeperAnalysis.json','w') as channel:
    json.dump(data_out,channel,indent=2)

            
            
            
