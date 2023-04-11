import numpy as np                              
import neal
import dimod
import math
import os
import pandas as pd
import matplotlib.pyplot as plt 
import json
import dwave.inspector
from functions import  get_qubo

from dwave.system import EmbeddingComposite
from dwave.system import DWaveSampler
from dwave.system import LeapHybridSampler


def getGndSequences(EXPERIMENT_IDX,HEATMAP,RUN,CYCLE):

    # --------------------------------------------------------------------------------------#
    # LOAD CURRENT EXPERIMENT SETTINGS;

    working_dir = os.path.join(f'heatmap_{HEATMAP}',f'run_{RUN}')
    filePath = os.path.join(working_dir, 'experiment.json')
    
    with open(filePath, 'r') as f:      
        full_data = json.load(f)
    data = full_data[EXPERIMENT_IDX]
    
    EXPERIMENT_NAME = data["NAME"]                  # Experiment name: will be the output folder;
    N_X = int(data["N_X"])                          # Width of the lattice containing the Hamiltonian path;
    N_Y = int(data["N_Y"])                          # Height of the lattice containing the Hamiltonian path;
                                                    #   amount of f-monomers  is fixed);
    S_IND = int(data["TARGET_STRUCTURE"])           # Index associated with the DESIGNABLE target structure;
    DICT_SIZE = int(data["DICT_SIZE"])              # Dictionary size;
    
    QA_ON = bool(data["QA_ON"])                     # Quantum Annealing;
    HQA_ON = bool(data["HQA_ON"])                   # Hybrid Quantum Annealing;
    

    # --------------------------------------------------------------------------------------#
    # CHOOSE PARAMETERS ASSOCIATED WITH ANNEALING PROCEDURES;

    N_SWEEPS_QA = int(1e2)                          # Quantum Annealing: number of annealing runs;
    ANNEALING_TIME = 200                            # Quantum Annealing: annealing time (from 10 to 2000);
    N_SWEEPS_HQA = 3                                # Hybrid Quantum Annealing: number of runs;

    sep = '\n---------------------------------------------------------------------------------\n'
    print(sep,f'GET GROUND-STATE SEQUENCES FOR EXPERIMENT {EXPERIMENT_IDX}: \t {EXPERIMENT_NAME}',sep)

    # --------------------------------------------------------------------------------------#
    # OUTPUT FOLDER
    
    cycle_folder = os.path.join(working_dir,f'cycle_{CYCLE}') 
   
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
    
    get_qubo(EXPERIMENT_IDX,HEATMAP,RUN,CYCLE,AVG_ON=False)
    Q = np.loadtxt('Q.txt')
    offset = np.loadtxt('offset.txt')
    bqm = dimod.BQM.from_qubo(Q)

    # --------------------------------------------------------------------------------------#
    # QA;
    if QA_ON:
        print('Quantum Annealing\n')
        sampler =  EmbeddingComposite(DWaveSampler())
        
        schedule = [[0,0],[ANNEALING_TIME*0.25,0.4],[ANNEALING_TIME*0.75,0.4],[ANNEALING_TIME,1]]
        chain_strength = np.max(abs(Q))
        bqmSampleSet = sampler.sample(bqm, num_reads=N_SWEEPS_QA, anneal_schedule=schedule,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_qa',chain_strength=chain_strength) 
        dwave.inspector.show(bqmSampleSet)
        bqmSampleSet = bqmSampleSet.aggregate()
    
        sampleSet_df = bqmSampleSet.to_pandas_dataframe(sample_column=True)  
        sampleSet_df = sampleSet_df.sort_values('energy',axis=0, ascending=True)

        file_json = os.path.join(cycle_folder,'qa.json')
        sampleSet_df.to_json(file_json,indent=2)
   
    # --------------------------------------------------------------------------------------#
    # HQA

    if HQA_ON:
        qpu_time = []
        run_time = []
        
        print('Hybrid quantum annealing\n')
        for n in range(N_SWEEPS_HQA):
            sampler = LeapHybridSampler()
            sampleSet = sampler.sample(bqm,label=f'{EXPERIMENT_NAME}_cycle_{CYCLE}_hqa',time_limit = 3)
            qpu_time.append(sampleSet.info['qpu_access_time'])
            run_time.append(sampleSet.info['run_time'])

            sampleSet_df = sampleSet.to_pandas_dataframe(sample_column=True) 
            sampleSet_df['energy']  = sampleSet_df['energy'] + offset

            file_json = os.path.join(cycle_folder,'hqa.json')

            if os.path.exists(file_json):
                df = pd.read_json(file_json)
                df = pd.concat([df,sampleSet_df],ignore_index=True)
                df = df.sort_values('energy',axis=0,ascending=True)
            else:
                df = sampleSet_df
            
            df.to_json(file_json, indent=2)
        
        np.savetxt(f'{cycle_folder}/qpu_time.txt',qpu_time)
        np.savetxt(f'{cycle_folder}/run_time.txt',run_time)
      
