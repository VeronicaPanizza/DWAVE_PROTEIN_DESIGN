import os
import random
import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display

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

# --------------------------------------------------------------------------------------#
    
def get_qubo(EXPERIMENT_IDX,HEATMAP,RUN,CYCLE,AVG_ON = False):

    with open('experiment.json','r') as channel:
        full_data = json.load(channel)
    data = full_data[EXPERIMENT_IDX]

    Nx   = int(data['N_X'])
    Ny   = int(data['N_Y'])
    D    = int(data['DICT_SIZE'])
    comp = data['COMPOSITION'][:-1]
    targ = int(data['TARGET_STRUCTURE'])

    d = Nx*Ny * (D-1)

    # Load contact map of target
    c_map = np.loadtxt(f'DATA/X_{Nx}_Y_{Ny}/contact_map_{targ}.txt')

    # Load average contact map of Nx x Ny lattice;
    avg_contact_map = np.zeros((Nx*Ny,Nx*Ny))
    if AVG_ON:
        avg_contact_map = np.loadtxt(f'DATA/X_{Nx}_Y_{Ny}/avg_contact_map.txt')
    c_map = c_map - avg_contact_map

    # Load energy map of system
    e_map_path = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}',f'cycle_{CYCLE}',f'dict_size_{D}.txt') 
    e_map = np.loadtxt(e_map_path,delimiter = ' ')

    def pos_q(i,m):
        return i * (D-1) + m 

    def q(vec,i,m):
        p_q = pos_q(i,m)
        return vec[p_q]

    l_exclud = 36 #2
    l_compos = 36 #2

    Q_energy = np.zeros((d,d),dtype='float')
    offset_energy = 0

    for i in range(Nx*Ny):
        for j in range(i+1,Nx*Ny):
            
            if np.around(c_map[i,j],4) != 0:
                
                offset_energy += e_map[D-1,D-1] * c_map[i,j]
                
                for m in range(D-1):
                    pos_qim = pos_q(i,m)
                    pos_qjm = pos_q(j,m)
                    
                    E = e_map[D-1,m] - e_map[D-1,D-1] 
                    
                    Q_energy[pos_qim,pos_qim] += E *c_map[i,j]
                    Q_energy[pos_qjm,pos_qjm] += E *c_map[i,j]
                    
                    for n in range(D-1):
                        pos_qjn = pos_q(j,n)

                        E = e_map[m,n]- e_map[D-1,n] - e_map[D-1,m] + e_map[D-1,D-1]
                        Q_energy[pos_qim,pos_qjn] += E * c_map[i,j]


    Q_exclud = np.zeros((d,d),dtype='float')
    for i in range(Nx*Ny):
        for m in range(D-1):
            for n in range(m+1, D-1):
                pos_qim = pos_q(i,m)
                pos_qin = pos_q(i,n)
                
                Q_exclud[pos_qim,pos_qin] += 1


    Q_compos = np.zeros((d,d),dtype='float')
    offset_compos = 0

    for m in range(D-1):
        offset_compos += comp[m]**2
        
        for i in range(Nx*Ny):
                pos_qim = pos_q(i,m)
                Q_compos[pos_qim,pos_qim] -= 2*comp[m] - 1

                for j in range(i+1,Nx*Ny):
                    pos_qjm = pos_q(j,m)
                    Q_compos[pos_qim,pos_qjm] += 2


    Q_energy = (Q_energy + Q_energy.transpose()) / 2
    Q_exclud = (Q_exclud + Q_exclud.transpose()) / 2
    Q_compos = (Q_compos + Q_compos.transpose()) / 2

    Q = Q_energy + l_exclud * Q_exclud + l_compos * Q_compos
    offset = offset_energy + l_compos * offset_compos

    np.savetxt('Q.txt',Q,delimiter = ' ')

def get_qubo_2(EXPERIMENT_IDX,e_map,AVG_ON = False):

    with open('experiment.json','r') as channel:
        full_data = json.load(channel)
    data = full_data[EXPERIMENT_IDX]

    Nx   = int(data['N_X'])
    Ny   = int(data['N_Y'])
    D    = int(data['DICT_SIZE'])
    comp = data['COMPOSITION'][:-1]
    targ = int(data['TARGET_STRUCTURE'])

    d = Nx*Ny * (D-1)

    # Load contact map of target
    c_map = np.loadtxt(f'DATA/X_{Nx}_Y_{Ny}/contact_map_{targ}.txt')

    # Load average contact map of Nx x Ny lattice;
    avg_contact_map = np.zeros((Nx*Ny,Nx*Ny))
    if AVG_ON:
        avg_contact_map = np.loadtxt(f'DATA/X_{Nx}_Y_{Ny}/avg_contact_map.txt')
    c_map = c_map - avg_contact_map

    def pos_q(i,m):
        return i * (D-1) + m 

    def q(vec,i,m):
        p_q = pos_q(i,m)
        return vec[p_q]

    l_exclud = 2
    l_compos = 2

    Q_energy = np.zeros((d,d),dtype='float')
    offset_energy = 0

    for i in range(Nx*Ny):
        for j in range(i+1,Nx*Ny):
            
            if np.around(c_map[i,j],4) != 0:
                
                offset_energy += e_map[D-1,D-1] * c_map[i,j]
                
                for m in range(D-1):
                    pos_qim = pos_q(i,m)
                    pos_qjm = pos_q(j,m)
                    
                    E = e_map[D-1,m] - e_map[D-1,D-1] 
                    
                    Q_energy[pos_qim,pos_qim] += E *c_map[i,j]
                    Q_energy[pos_qjm,pos_qjm] += E *c_map[i,j]
                    
                    for n in range(D-1):
                        pos_qjn = pos_q(j,n)

                        E = e_map[m,n]- e_map[D-1,n] - e_map[D-1,m] + e_map[D-1,D-1]
                        Q_energy[pos_qim,pos_qjn] += E * c_map[i,j]


    Q_exclud = np.zeros((d,d),dtype='float')
    for i in range(Nx*Ny):
        for m in range(D-1):
            for n in range(m+1, D-1):
                pos_qim = pos_q(i,m)
                pos_qin = pos_q(i,n)
                
                Q_exclud[pos_qim,pos_qin] += 1


    Q_compos = np.zeros((d,d),dtype='float')
    offset_compos = 0

    for m in range(D-1):
        offset_compos += comp[m]**2
        
        for i in range(Nx*Ny):
                pos_qim = pos_q(i,m)
                Q_compos[pos_qim,pos_qim] -= 2*comp[m] - 1

                for j in range(i+1,Nx*Ny):
                    pos_qjm = pos_q(j,m)
                    Q_compos[pos_qim,pos_qjm] += 2


    Q_energy = (Q_energy + Q_energy.transpose()) / 2
    Q_exclud = (Q_exclud + Q_exclud.transpose()) / 2
    Q_compos = (Q_compos + Q_compos.transpose()) / 2

    Q = Q_energy + l_exclud * Q_exclud + l_compos * Q_compos
    offset = offset_energy + l_compos * offset_compos
    print('Offset:',offset)
    np.savetxt('Q.txt',Q,delimiter = ' ')

# --------------------------------------------------------------------------------------#

def get_drawing(RUN,CYCLE):
    EXPERIMENT_IDX = 5

    with open('experiment.json','r') as channel:
        full_data = json.load(channel)
    data = full_data[EXPERIMENT_IDX]

    name = f'heatmap_{RUN}/run_{RUN}'
    targ = int(data['TARGET_STRUCTURE'])
    Nx   = int(data['N_X'])
    Ny   = int(data['N_Y'])
    comp = np.array(data['COMPOSITION'])
    D    = int(data['DICT_SIZE'])

    colors = ['red','blue','yellow','green','magenta']
    colors = colors[: D]

    def pos_q(i,m):
        return i * (D-1) + m

    def q(vec,i,m):
        return vec[pos_q(i,m)]

    def get_colors(vec):
        out_colors = []

        for i in range(Nx*Ny):

            color_on = False
            for m in range(D-1):
                if q(vec,i,m) == 1 and not color_on:
                    color_on = True
                    out_colors.append(colors[m])
                elif q(vec,i,m) == 1 and color_on:
                    raise Exception('Excluded volume condition violated')

            if not color_on:
                out_colors.append(colors[D-1])
        return out_colors

    # Load data
    file_path = os.path.join(name,f'cycle_{CYCLE}','hqa.json')
    df = pd.read_json(file_path)
    display(df)

    for n in range(3):
        state  = list(df['sample'][n].values())
        state = np.array(state)
        energy = df['energy'][n]

        # Decode colors;
        colors = get_colors(state)
        print(f'Energy: {energy}')

        # x = [0, 0, 1, 2, 3, 3, 2, 1, 1, 0, 0, 1, 2, 2, 3, 3]
        # y = [2, 3, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 0]
        x = [1,1,2,2,3,3,4,4,5,5,5,4,3,2,1,0,0,0,0,1,1,0,0,1,2,3,3,2,2,3,4,5,5,5,4,4]
        y = [3,4,4,3,3,4,4,3,3,4,5,5,5,5,5,5,4,3,2,2,1,1,0,0,0,0,1,1,2,2,2,2,1,0,0,1]

        figure = plt.figure(figsize=(6,6))
        plt.grid()
        struct = plt.plot(y,x,color ='k')
        for color, xp,yp in list(zip(colors,x,y)):
            plt.plot(yp,xp,linestyle='',marker = 'o',markerfacecolor=color,color=color)
        figure.savefig(f'conf_{n}')

get_drawing(0,5)