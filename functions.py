import os
import random
import numpy as np

# --------------------------------------------------------------------------------------#

def get_eMap(DICT_SIZE):
    random.seed()    
    eMap = np.random.random((DICT_SIZE,DICT_SIZE)) - .5 
    eMap = eMap + eMap.transpose() 
    eMap = eMap / np.linalg.norm(eMap)
    
    return eMap

# --------------------------------------------------------------------------------------#

def get_cPot(DICT_SIZE):
    random.seed()
    cPot = np.random.rand(1,DICT_SIZE)
    filePath = os.path.join('CHEMICAL_POTENTIAL',f'dict_size_{DICT_SIZE}')
    if os.path.exists(filePath):
        print('Impossible to generate new chemical potential, one already exists')
    else:
        print('Chemical potential correctly generated')
        np.savetxt(filePath,cPot,fmt='%.3f')

# --------------------------------------------------------------------------------------#

def get_chainInt(DICT_SIZE):
    random.seed()
    chainInt = np.random.rand(DICT_SIZE,DICT_SIZE)
    chainInt = 0.5*(chainInt + chainInt.transpose())
    filePath = os.path.join('CHAIN_BONDS_MAP',f'dict_size_{DICT_SIZE}')
    if os.path.exists(filePath):
        print('Impossible to generate new chain interaction map, one already exists')
    else:
        print('Chain interaction map correctly generated')
        np.savetxt(filePath,chainInt,fmt='%.3f')

# --------------------------------------------------------------------------------------#

def n(d1,d2,c_map,sequence):
    
    #   INPUTS:
    #       d1:       first monomer type;
    #       d2:       second monomer type;
    #       c_map:    contact map of the structure;
    #       sequence: sequence of monomer types;
    
    #   OUTPUT:
    #       Number of (d1, d2) contacts in the folded protein;
    
    pos1 = [x for x in range(len(sequence)) if sequence[x] == d1]
    pos2 = [x for x in range(len(sequence)) if sequence[x] == d2]
    
    n = 0
    for p1 in pos1:
        for p2 in pos2:
            if c_map[p1,p2] == 1:
                n = n + 1
    if d1 == d2:
        n = n / 2    
    return n

# --------------------------------------------------------------------------------------#

def n_vector(sequence,contactMaps,DICT_SIZE):
    # sequence with dimension        = (#sequences, length);
    # contactMaps with dimension     = (#structures, length, length);
    
    vector = np.zeros((np.shape(contactMaps)[0],np.shape(sequence)[0],DICT_SIZE , DICT_SIZE))
    for gamma_idx, gamma in enumerate(contactMaps):
        for seq_idx, seq in enumerate(sequence):
            for m1 in range(DICT_SIZE):
                for m2 in range(m1,DICT_SIZE):
                    vector[gamma_idx,seq_idx,m1,m2] = n(m1,m2,gamma,seq)

    vector = (vector + np.transpose(vector, (0,1,3,2))) / 2   

    # Output shape: (#structures, #sequences, DICT_SIZE ** 2)
    vector = np.reshape(vector,([np.shape(vector)[0],np.shape(vector)[1],-1]))  

    return vector