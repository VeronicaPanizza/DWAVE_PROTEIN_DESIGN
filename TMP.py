import os
from MAIN import MAIN
from getCorrectedEnergyMap import getCorrectedEnergyMap

for run in range(1):
    MAIN()
    os.rename('Small_lattice_1',f'Small_lattice_1_run{run}')
    os.rename('dist.png',f'matrix_convergence_run{run}.png')
    os.rename('distAnalysis.json',f'distAnalysis_run{run}.json')


