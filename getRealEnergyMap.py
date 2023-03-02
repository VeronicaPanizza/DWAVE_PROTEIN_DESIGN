import os
import numpy as np 
import json
from random import randint
import pandas as pd
import functions 

EXPERIMENT_IDX = 3


# --------------------------------------------------------------------------------------#
# LOAD CURRENT EXPERIMENT SETTINGS;
    
with open('experiment.json', 'r') as f:      
    full_data = json.load(f)
data = full_data[EXPERIMENT_IDX]

N_X             = int(data["N_X"])                  # Width of the lattice containing the Hamiltonian path;
N_Y             = int(data["N_Y"])                  # Height of the lattice containing the Hamiltonian path;
DICT_SIZE       = int(data["DICT_SIZE"])            # Dictionary size;

N_SEQUENCES     = int(1e3) 
N_ENERGY_MAPS   = int(1e3)    
TEMPERATURE     = 0.15323                           # Temperature is in range (0,1);

# --------------------------------------------------------------------------------------#
# LOAD ENUMERATED STRUCTURES;

folder = os.path.join('DATA/',f'X_{N_X}_Y_{N_Y}/')
dir_content = os.listdir(folder)
dir_content_filtered = [dir_content[x] for x in range(len(dir_content)) \
    if 'contact_map' in dir_content[x]] 

N_STRUCTURES = len(dir_content_filtered)

contactMaps = np.zeros((N_STRUCTURES,N_X * N_Y,N_X * N_Y))
for file_idx,file in enumerate(dir_content_filtered):
    file = os.path.join(folder,file)
    contactMaps[file_idx,:,:] = np.loadtxt(file, delimiter = ' ')

# --------------------------------------------------------------------------------------#
# CAMPION RANDOMLY 'N_SEQUENCES';

sequences = -np.ones((N_SEQUENCES,N_X * N_Y))
for s_idx in range(N_SEQUENCES):
    for site in range(N_X * N_Y):
        sequences[s_idx,site] = randint(0,DICT_SIZE-1)
        

n = functions.n_vector(sequences,contactMaps,DICT_SIZE)
n = n[np.newaxis,:,:]

# --------------------------------------------------------------------------------------#
# GENERATE ENERGY MAPS PROPOSALS;

eMaps = np.zeros((N_ENERGY_MAPS,1,1,DICT_SIZE**2))
for e_idx in range(N_ENERGY_MAPS):
    e = np.random.random((DICT_SIZE,DICT_SIZE)) - .5 
    e = e + e.transpose()
    e = e / np.linalg.norm(e)
    e = np.reshape(e,(1,-1))
    eMaps[e_idx,0,0,:] = e

# --------------------------------------------------------------------------------------#
# EVALUATE ENERGY;
# For every sequence, we can calculate its spectrum across the configuration space;

# eMaps     = (N_ENERGY_MAPS, 1           , 1          , DICT_SIZE**2)
# n         = (1            , N_STRUCTURES, N_SEQUENCES, DICT_SIZE**2)

energy = np.sum(eMaps * n, axis=3)
energy = np.transpose(energy,(0,2,1))       # Shape: (N_ENERGY_MAPS, N_SEQUENCES, N_STRUCTURES);

outputData = {}

for eMap_idx in range(N_ENERGY_MAPS):
    eMap_name = f'map_{eMap_idx}'
    outputData[eMap_name] = {}
    for s_idx in range(N_STRUCTURES):
        s_name = f'structure_{s_idx}'
        outputData[eMap_name][s_name] = list()

for eMap_idx in range(N_ENERGY_MAPS):
    eMap_name = f'map_{eMap_idx}'
    eMap = eMaps[eMap_idx]

    ENERGY_GAP = TEMPERATURE * (N_X - 1) * (N_Y - 1) * (np.max(eMap)-np.min(eMap)) 

    for s_idx in range(N_SEQUENCES):
        gndEnergy = min(energy[eMap_idx,s_idx,:])   
        gndStructure = np.where(energy[eMap_idx,s_idx,:] == gndEnergy)[0]
        
        if len(gndStructure) == 1:
            gndStructure = gndStructure[0]
            firstEnergy = min([energy[eMap_idx,s_idx,x] \
                for x in range(N_STRUCTURES) if x != gndStructure])

            energy_gap = firstEnergy - gndEnergy

            if energy_gap >= ENERGY_GAP:
                structure_name = f'structure_{gndStructure}'

                outputData[eMap_name][structure_name].append(sequences[s_idx])                

designability = np.zeros(len(outputData))

for m_idx,m in enumerate(outputData):
    designability[m_idx] = 0

    for structure in outputData[m]:
        
        designability[m_idx] += len(outputData[m][structure])

print('\n Number of sequences that uniquely fold:',designability)

eMap_idx = np.where(designability == max(designability))[0]
eMap_idx = eMap_idx[0]
eMap = eMaps[eMap_idx,0,0,:].reshape((DICT_SIZE,DICT_SIZE))
fileName = os.path.join('ENERGY_MAP',f'dict_size_{DICT_SIZE}.txt')
np.savetxt(fileName,eMap,fmt = '%.5f',delimiter='\t')

outputDataFrame = pd.DataFrame.from_dict(outputData)
outputDataFrame.to_json('output.json',indent=2)

print(f'\n Maximal number of sequences that uniquely fold: {max(designability)} with energy map: {eMap_idx}')

eMaps = eMaps.squeeze(1)
eMaps = eMaps.squeeze(1)
np.savetxt('eMaps.txt',eMaps)