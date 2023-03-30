import os
from MAIN import MAIN
from getCorrectedEnergyMap import getCorrectedEnergyMap
from functions import get_qubo

EXPERIMENT_IDXS=[5]

for run in range(4):
    MAIN(EXPERIMENT_IDXS)
    os.rename('ExtraLarge_lattice',f'ExtraLarge_lattice_run{run}')
    os.rename('dist.png',f'matrix_convergence_run{run}.png')
    os.rename('distAnalysis.json',f'distAnalysis_run{run}.json')


