import numpy as np                              
import neal
import dimod
import math
import os
import pandas as pd
import matplotlib.pyplot as plt 
import json
import dwave.inspector

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
    
    lambdaCons          = 2.2                        # Energy penality for putting two flavours in the same vertex;
    lambdaComposition   = 1.5

    # --------------------------------------------------------------------------------------#
    # CHOOSE PARAMETERS ASSOCIATED WITH ANNEALING PROCEDURES;

    N_SWEEPS_SA = int(500)                          # Simulated Annealing: number of annealing runs;
    N_STEPS_SA = 1e5                                # Simulated Annealing: number of steps per run;
    N_SWEEPS_QA = int(1e2)                          # Quantum Annealing: number of annealing runs;
    ANNEALING_TIME = 2000                           # Quantum Annealing: annealing time (from 10 to 2000);
    N_SWEEPS_HQA = 3                                # Hybrid Quantum Annealing: number of runs;

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,f'GET GROUND-STATE SEQUENCES FOR EXPERIMENT {EXPERIMENT_IDX}: \t {EXPERIMENT_NAME}',sep)

    # --------------------------------------------------------------------------------------#
    # OUTPUT FOLDER
    
    cycle_folder = os.path.join(EXPERIMENT_NAME,f'cycle_{CYCLE}') 
   
    # --------------------------------------------------------------------------------------#
    # Create or load energy map;
                           
    filePath = os.path.join(cycle_folder, f'dict_size_{DICT_SIZE}.txt') 
    eMap = np.loadtxt(filePath)

    print(f'\n Cycle: \t {CYCLE}')
    print(f'\n Current energy map: \n{eMap}')
    
    # Load contact matrix
    filePath = os.path.join(f'DATA/X_{N_X}_Y_{N_Y}', f'contact_map_{S_IND}.txt')
    contacts = np.loadtxt(filePath)
    nTotalSites = np.size(contacts,axis=0)

    print('\nTotal number of sites:\t\t %i'% nTotalSites)

    # --------------------------------------------------------------------------------------#
    # SETUP OPTIMIZATION PROBLEM;

    q = [[dimod.Binary(label=f'q_{s}_{t}') for t in range(DICT_SIZE)] for s in range(nTotalSites)]                                                                  

    energy = 0
    
    # Impose constraints on each site;
    for s in range(nTotalSites):
        sum_tot = 0
        for m in range(DICT_SIZE):
            sum_tot += q[s][m]
        energy += lambdaCons * (sum_tot - 1)**2
    
    # Contribution of contacts;
    for s1 in range(nTotalSites-1):
        for s2 in range(s1+1,nTotalSites):
            if contacts[s1,s2] == 0:
                continue
            else:
                for m1 in range(DICT_SIZE):
                    for m2 in range(DICT_SIZE):
                        energy += eMap[m1,m2] * q[s1][m1] * q[s2][m2]

    # Composition constraint;
    sum_composition = 0
    for m in range(DICT_SIZE):
        sum_flavor = 0
        for s in range(nTotalSites):
            sum_flavor += q[s][m]
        sum_composition += (sum_flavor - COMPOSITION[m])**2
    energy += lambdaComposition * sum_composition

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
        
        bqmSampleSet = sampler.sample(bqm, num_reads=N_SWEEPS_QA, anneal_schedule=schedule,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_qa',chain_strength=10) 
        dwave.inspector.show(bqmSampleSet)
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
            sampleSet = sampler.sample(bqm,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_hqa',time_limit = 3)
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
        
      