#!/bin/bash
#SBATCH --job-name=BipedalWalkerNEAT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --time=2:00:00
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
python ~/NEAT_python/src/set_train_config/set_train_config_BipedalWalker-v3.py
python ~/NEAT_python/src/evolution/evolving/evolve_BipedalWalker.py