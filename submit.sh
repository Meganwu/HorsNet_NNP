#!/bin/bash                                                             
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH -p  gpu

module load gcc
module load  cuda/11.3.1

python   spooky_0211_cuda.py 
 
echo "Nequip trained."






