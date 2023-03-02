import numpy as np                              
import neal
import dimod
import math
import os
import pandas as pd
import matplotlib.pyplot as plt 
import json

from dwave.system import EmbeddingComposite
from dwave.system import DWaveSampler
from dwave.system import LeapHybridSampler


def getGndSequences(EXPERIMENT_IDX,CYCLE):

    # --------------------------------------------------------------------------------------#
    # LOAD CURRENT EXPERIMENT SETTINGS;

    with open('experiment.json', 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]
    
    EXPERIMENT_NAME = data["NAME"]                  # Experiment name: will be the output folder;
    N_X = int(data["N_X"])                          # Width of the lattice containing the Hamiltonian path;
    N_Y = int(data["N_Y"])                          # Height of the lattice containing the Hamiltonian path;
    COMPOSITION = data["COMPOSITION"]               # Composition constraint on sequence (for each flavour - f, the 
                                                    #   amount of f-monomers  is fixed);
    S_IND = int(data["TARGET_STRUCTURE"])           # Index associated with the DESIGNABLE target structure;
    DICT_SIZE = int(data["DICT_SIZE"])              # Dictionary size;
    VANILLA_SA_ON = bool(data["SA_ON"])             # Simulated Annealing (no schedule choice);
    QA_ON = bool(data["QA_ON"])                     # Quantum Annealing;
    HQA_ON = bool(data["HQA_ON"])                   # Hybrid Quantum Annealing;
    
    lambdaCons   = 5                                # Energy penality for putting two flavours in the same vertex;
    lambdaSlk    = 5                                # Energy penality for system states that are incompatible with slack variables' values; 
    lambdaSlkTot = 4                                # Energy penality for system with a wrong number of particles (used only to enforce condition 1?);
    # epsilon      = 1                              # Energy penality to break the geometric symmetry (reducing the groundspace degeneration);

    CONTACT_INTERACTION_ON = True
    
    # --------------------------------------------------------------------------------------#
    # CHOOSE PARAMETERS ASSOCIATED WITH ANNEALING PROCEDURES;

    N_SWEEPS_SA = int(500)                          # Simulated Annealing: number of annealing runs;
    N_STEPS_SA = 1e5                                # Simulated Annealing: number of steps per run;
    N_SWEEPS_QA = int(1e2)                          # Quantum Annealing: number of annealing runs;
    ANNEALING_TIME = 2000                           # Quantum Annealing: annealing time (from 10 to 2000);
    N_SWEEPS_HQA = 10                               # Hybrid Quantum Annealing: number of runs;

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,f'GET GROUND-STATE SEQUENCES FOR EXPERIMENT {EXPERIMENT_IDX}: \t {EXPERIMENT_NAME}',sep)

    # --------------------------------------------------------------------------------------#
    # OUTPUT FOLDER
    
    cycle_folder = os.path.join(EXPERIMENT_NAME,f'cycle_{CYCLE}') 
   
    # --------------------------------------------------------------------------------------#
    # Create or load energy map (NEEDS ADAPTATION);
                           
    filePath = os.path.join(cycle_folder, f'dict_size_{DICT_SIZE}.txt') 
    eMap = np.loadtxt(filePath)

    print(f'\n Cycle: \t {CYCLE}')
    print(f'\n Current energy map: \n{eMap}')
    
    # Load contact matrix
    filePath = os.path.join(f'DATA/X_{N_X}_Y_{N_Y}', f'contact_map_{S_IND}.txt')
    contacts = np.loadtxt(filePath)
    nTotalSites = np.size(contacts,axis=0)

    # Load simple paths;
    filePath = os.path.join(f'DATA/X_{N_X}_Y_{N_Y}',f'simple_paths_{S_IND}.txt')
    simple_paths = list()
    with open(filePath,'r') as fp:
        [simple_paths.append(np.fromstring(x,dtype=int,sep=' ')) for x in fp.readlines()]
    
    # Number of sites in contact;
    contactSites = list()
    s = 0
    for line in contacts:
        if sum(line) > 0:
            contactSites.append(s)
        s += 1

    nContactSites = len(contactSites)
    nDisconnectedSites = nTotalSites - nContactSites
    print('\nTotal number of sites:\t\t %i'% nTotalSites)
    print('Number of sites in contact:\t %i' % nContactSites)
    print('Number of disconnected sites:\t %i\n' % nDisconnectedSites)

    # --------------------------------------------------------------------------------------#
    # SETUP OPTIMIZATION PROBLEM;

    q = [[dimod.Binary(label=f'q_{contactSites[s]}_{t}') for t in range(DICT_SIZE)] for s in range(nContactSites)]                                                                  

    composition = np.zeros((DICT_SIZE, nDisconnectedSites+1))
    for t in range(DICT_SIZE):
        for n in range(-nDisconnectedSites, 1):
            composition[t,n] = COMPOSITION[t] + n
    composition = composition.astype(int)


    slk = [[dimod.Binary(label=f'slk_{t}_{n}') for n in composition[t,:]]for t in range(DICT_SIZE)]
    energy = 0

    # Impose constraints on each site;
    for s in range(nContactSites):
        sum_tot = 0
        for m in range(DICT_SIZE):
            sum_tot += q[s][m]
        energy += lambdaCons * (sum_tot - 1)**2
    
    # Impose composition constraints;
    for t in range(DICT_SIZE):
        sum_flavour = 0
        for s in range(nContactSites):
            sum_flavour += q[s][t]  
        sum_slack = 0          
        for n in range(nDisconnectedSites +1):
            sum_slack += composition[t,n] * slk[t][n]    
        energy += lambdaSlk * (sum_slack - sum_flavour)**2

    
    for t in range(DICT_SIZE):
        sum_tot = 0
        for n in range(nDisconnectedSites+1):
            sum_tot += slk[t][n]
        energy += lambdaSlkTot * (sum_tot - 1)**2  
    
    
    # Contribution of contacts;
    if CONTACT_INTERACTION_ON:    
        
        for s1 in range(nContactSites-1):
            for s2 in range(s1+1,nContactSites):
                if contacts[contactSites[s1],contactSites[s2]] == 0:
                    continue
                else:
                    for m1 in range(DICT_SIZE):
                        for m2 in range(DICT_SIZE):
                            energy += eMap[m1,m2] * q[s1][m1] * q[s2][m2]
    '''
    # Adding a term that explicitly breaks the geometric symmetry;
    for s in simple_paths:
        max_depth = math.floor(len(s)/2)
        for depth in range(max_depth):
            s1, s2 = s[depth], s[len(s)-1-depth]
            for m1 in range(DICT_SIZE-1):
                for m2 in range(m1+1,DICT_SIZE):
                    idx1, idx2 = contactSites.index(s1),contactSites.index(s2)
                    energy += (epsilon / pow(3,depth)) * q[idx2][m1] * q[idx1][m2]
    '''
    
    bqm = dimod.BinaryQuadraticModel(energy)
    
    # --------------------------------------------------------------------------------------#
    # VANILLA SA;
    
    if VANILLA_SA_ON:
        sampler = neal.SimulatedAnnealingSampler()
        bqmSampleSet = sampler.sample(bqm,num_reads=N_SWEEPS_SA,num_sweeps=N_STEPS_SA,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_sa')

        # Output simulated annealing solution;
        print('Simulated Annealing (No schedule and range for beta)\n')
        print('Number of steps SA:\t %.0e' % N_STEPS_SA)
        print('SA number of reads SA:\t %.0e' % N_SWEEPS_SA)
        bqmSampleSet = bqmSampleSet.aggregate()
        sampleSet_df = bqmSampleSet.to_pandas_dataframe(sample_column=True)  
        sampleSet_df = sampleSet_df.sort_values('energy',axis=0, ascending=True,ignore_index=True)

        file_json = os.path.join(cycle_folder,'sa.json')
        sampleSet_df.to_json(file_json,indent=2)

    # --------------------------------------------------------------------------------------#
    # QA;
    if QA_ON:
        print('Quantum Annealing\n')
        sampler =  EmbeddingComposite(DWaveSampler())
        
        schedule = [[0,0],[ANNEALING_TIME,1]]
        
        bqmSampleSet = sampler.sample(bqm, num_reads=N_SWEEPS_QA, anneal_schedule=schedule,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_qa') 
        bqmSampleSet = bqmSampleSet.aggregate()
    
        sampleSet_df = bqmSampleSet.to_pandas_dataframe(sample_column=True)  
        sampleSet_df = sampleSet_df.sort_values('energy',axis=0, ascending=True)

        file_json = os.path.join(cycle_folder,'qa.json')
        sampleSet_df.to_json(file_json,indent=2)
    # --------------------------------------------------------------------------------------#
    # HQA

    if HQA_ON:
        print('Hybrid quantum annealing\n')
        for n in range(N_SWEEPS_HQA):
            sampler = LeapHybridSampler()
            sampleSet = sampler.sample(bqm,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_hqa')
            sampleSet_df = sampleSet.to_pandas_dataframe(sample_column=True) 
            
            file_json = os.path.join(cycle_folder,'hqa.json')

            if os.path.exists(file_json):
                df = pd.read_json(file_json)
                df = pd.concat([df,sampleSet_df],ignore_index=True)
                df = df.sort_values('energy',axis=0,ascending=True)
            else:
                df = sampleSet_df
            
            # Aggregate samples;
            samples = dimod.as_samples(list(df["sample"]))                    
            energies = np.array(df["energy"])
            num_occurrences = np.array(df["num_occurrences"])

            df = dimod.SampleSet.from_samples(samples_like=samples,vartype='BINARY',energy=energies,num_occurrences=num_occurrences,aggregate_samples=True)
            df = df.to_pandas_dataframe(sample_column = True)
            df.to_json(file_json, indent=2)
        
      