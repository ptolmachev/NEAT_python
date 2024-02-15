#!/bin/bash
#SBATCH --job-name=PendulumNEAT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --time=5:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load anaconda3/2022.5

# Activate conda environment
conda activate NEAT_environment
python ~/NEAT_python/src/evolution/evolving/evolve.py