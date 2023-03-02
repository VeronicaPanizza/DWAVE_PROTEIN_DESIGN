import os
import numpy as np 
from numpy import linalg
import pandas as pd
import json
from IPython.display import display

EXPERIMENT_IDX = 0
TEMPERATURE = 0.15323

with open('experiment.json', 'r') as f:      
    full_data = json.load(f)
data = full_data[EXPERIMENT_IDX]

EXPERIMENT_NAME = data['NAME']
DICT_SIZE       = data['DICT_SIZE']
N_X             = data['N_X']
N_Y             = data['N_Y']
S_IND           = data['TARGET_STRUCTURE']
MAX_CYCLES      = data['MAX_CYCLES']

fp = os.path.join('ENERGY_MAP',f'dict_size_{DICT_SIZE}.txt')
eMapReal = np.loadtxt(fp)
energy_gap_real = TEMPERATURE * (np.max(eMapReal) - np.min(eMapReal)) * (N_X - 1) * (N_Y - 1)
print('Real energy gap',energy_gap_real)

# LOADING RESULTS OF FOLDING PROCESS;
for RUN in range(1):
    Map ={}
        
    for CYCLE in range(7):
        fd = os.path.join(f'{EXPERIMENT_NAME}_run{RUN}',f'cycle_{CYCLE}')

        fp = os.path.join(fd,f'dict_size_{DICT_SIZE}.txt')
        entryName = f'cycle_{CYCLE}'
        tmp_map = np.loadtxt(fp, delimiter=' ')
        if np.sum(tmp_map) < 0:
            tmp_map = tmp_map - 2 * np.sum(tmp_map) / 9 
        Map[entryName] = tmp_map

        fp = os.path.join(fd,'gnd_structures_from_hqa.json')

        if not os.path.isfile(fp):
            print('End of deeper analysis')
            break 
        else:
            print('\n\n CYCLE:', CYCLE,'\n')


        df = pd.read_json(fp)
        df = df.sort_values(['energy'])
        df = df.reset_index(drop = True)
        sequences = np.vstack(df['sequence'])                                           # Extract all sequences from folding database
        r_sequences, r_idxs = np.unique(sequences, axis = 0, return_inverse=True)       # Remove redundant sequences (keeping track of
                                                                                        # the original array)

        SEQ = []                                                                        # List that will contain sequences that uniquely
                                                                                        # fold;
        GAP = []                                                                        

        GND_ENERGY = []                                                                 # Energy of sequences folded in their native
                                                                                        # structure;
        FIRST_POS = []                                                                  # Position of the target structure in the energy
                                                                                        # spectra for each sequence;

        for seq_idx in range(len(r_sequences)):

            idxs = [x for x in range(len(r_idxs)) if r_idxs[x] == seq_idx]              # Select indeces in database associated with a given sequence;
            r_df = df[df.index.isin(idxs)]                                              # Isolate sub-dataframe s.t. sequence is fixed; 
            r_df = r_df.sort_values(['energy'])
            r_df = r_df.reset_index(drop=True)
        
            if len(r_df) > 1:                                                           # Evaluate energy gap between groundstate and 
                gap = r_df['energy'][1] - r_df['energy'][0]                             # first excited level;
            else:
                raise Warning('Something went wrong')

            r_df = r_df[r_df['structure']==S_IND]                                       # Select sub-dataset s.t. the structure is the
                                                                                        # target one;

            first_pos = -1
            if len(r_df) != 0:
                first_pos = min(r_df.index)                                             # Select position of target structure in
                                                                                        # energy spectra;

            if gap > energy_gap_real:                                                   # If the sequence allows a native state, save
                                                                                        # associated data;
                FIRST_POS.append(first_pos)
                GAP.append(gap)
                SEQ.append(seq_idx)
                GND_ENERGY.append(min(r_df['energy']))

        data= {}                                                                        # Gather this data in a dictionary;
        data['Sequence_id'] = []
        data['Sequence'] = []
        data['Energy_gap'] = []
        data['Gnd_energy'] = []
        data['Target_position'] = []

        for s, g, e, p in zip(SEQ,GAP,GND_ENERGY,FIRST_POS):
            data['Sequence_id'].append(s)
            data['Sequence'].append(np.array(r_sequences[s]).astype(int))
            data['Energy_gap'].append(g)
            data['Gnd_energy'].append(e)
            data['Target_position'].append(p)

        fold_df = pd.DataFrame.from_dict(data)                                          # Convert the dictionary in a dataframe;
        fold_df = fold_df.sort_values('Gnd_energy')
        fold_df = fold_df.reset_index(drop=True)
        display(fold_df[:20])

        foldable_percentage = len(fold_df)/len(r_sequences) * 100

        t_fold_df = fold_df[fold_df['Target_position']==0]
        if len(fold_df)>0:
            target_foldable_percentage = len(t_fold_df) / len(fold_df) * 100
        else:
            target_foldable_percentage = 0
            
        if RUN == 0 and CYCLE == 0:
            data = {}
            data['run'] = []
            data['cycle'] = []
            data['foldable_percentage'] = []
            data['target_foldable_percentage'] = []

            data = pd.DataFrame.from_dict(data)
        else:
            data = pd.read_json('deeperAnalysis.json')

        data_add = {}
        data_add['run'] = [RUN]
        data_add['cycle'] = [CYCLE]
        data_add['foldable_percentage'] = [foldable_percentage]
        data_add['target_foldable_percentage'] = [target_foldable_percentage]

        data_add = pd.DataFrame.from_dict(data_add)
        data = pd.concat((data,data_add),ignore_index=True)
        data.to_json('deeperAnalysis.json',indent = 2)

    delta       = {}
    difference  = {}

    for CYCLE in range(MAX_CYCLES):
        entryName = f'cycle_{CYCLE}'
        if entryName in Map:
            delta[entryName] = linalg.norm(Map[entryName]-eMapReal)
        else:
            break
    
    for CYCLE in range(MAX_CYCLES):
        entryName = f'cycle_{CYCLE}'
        nextEntryName = f'cycle_{CYCLE+1}'
        name = f'step_{CYCLE}'
        if nextEntryName in Map:
            difference[name] = linalg.norm(Map[entryName]-Map[nextEntryName],2) 
        else:
            difference[name] = 'Nan'

    data_add = {}
    data_add['delta'] = []
    data_add['difference'] = []
    data_add['run'] = []
    data_add['cycle'] = []
    
    for c,(val1,val2) in enumerate(zip(difference,delta)):
        data_add['cycle'].append(c)
        data_add['run'].append(RUN)
        data_add['difference'].append(difference[val1])
        data_add['delta'].append(delta[val2])
    
    data_add = pd.DataFrame.from_dict(data_add)

    fp = 'matrix_convergence.json'

    if os.path.isfile(fp):
        data = pd.read_json(fp)
        data = pd.concat((data,data_add),ignore_index=True)
        data.to_json('matrix_convergence.json',indent=2)
    else:
        data_add.to_json('matrix_convergence.json',indent=2)






    

