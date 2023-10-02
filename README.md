# DWAVE_PROTEIN_DESIGN

CONTENT OF THE PROJECT:
- DATA folder        : contains hamiltonian paths for fixed size lattice;
- ENERGY_MAP folder  : contains energy maps that are retained as "physical"
- Start_emaps folder : contains energy maps that can be used as seeds to initialize the algorithm;
- experiment.json    : to be edited, initializes details concerning the simulation;
- TMP.py             : is the master python file, the one that is meant to be runned 


  TMP.py calls MAIN.py --> calls N_CYCLE times (getGndSequences.py -> getGndStructures.py -> getCorrectedEnergyMap.py)
