#!/bin/bash
#SBATCH -J ZDT3_qpots_dec_10_reps

#SBATCH --output=Output_Wrapper.out

#SBATCH -e Error_Wrapper.error

#SBATCH -N 1

#SBATCH --ntasks=10

#SBATCH --mem-per-cpu=8G

#SBATCH -t 12:00:00

###############SBATCH --mem=80GB

#SBATCH -A akr6198

#SBATCH -p sla-prio

#SBATCH --mail-type=ALL

#SBATCH --mail-user=peb5234@psu.edu



echo "Job starting on `hostname` at `date`"



echo -e "Slurm job ID: $SLURM_JOBID"



cd $SLURM_SUBMIT_DIR



# A little useful information for the log file...

echo -e "Master process running on: $HOSTNAME"

echo -e "Directory is: $PWD"



# Load the su2 module. The su2-intel module also loads all of the Intel suite

module purge

#module load cuda/11.5.0

#module load anaconda/2021.11
#module load anaconda3/2021.05

#module load cmake/3.21.4
#module load cmake/3.25.2-intel-2021.4

#module load intel/2021.4.0

#module load impi/2021.4.0

#module load tbb/2021.4.0

#module load mkl/2021.4.0

module load su2/8.0.1-intel
module use /storage/icds/RISE/sw8/modules
module load anaconda/2023.09
module load intel/2021.4.0
module load openmpi/4.1.1

#module load anaconda3/2021.05

#module load cmake/3.25.2-intel-2021.4

#source activate base

echo " "

echo "The following modules are in use"

module list

echo " "



# Call srun to launch the MPI-based job


#srun $SU2_RUN/SU2_GEO inv_ONERAM6_adv.cfg
#shape_optimization.py -n 48 -g CONTINUOUS_ADJOINT -o SLSQP -f  inv_NACA0012_basic_modified.cfg
#set_ffd_design_var.py -i 10 -j 1 -k 0 -b MAIN_BOX -m 'airfoil' --dimension 2
#srun $SU2_RUN/SU2_DEF inv_NACA0012_basic_modified.cfg
#python SU2_Wrapper_CD_Gradient_Only.py -mcfg config_NACA0012.cfg -mesh mesh_NACA0012.su2 -dv DV_VALUE.npy -np 240 -si 0
#python Gradient_Optimization_SLSQP.py
#python SU2_Wrapper_CD_Gradient_Only_2.py -mcfg "config_NACA0012.cfg" -mesh "mesh_NACA0012.su2" -dv "DV_VALUE.npy" -np 48
#/usr/bin/time -v parallel_computation.py -n 4 -f inv_ONERAM6_adv.cfg
#python SU2_Wrapper_All_Gradients_RANS.py -mcfg "ffd_rae2822_4pts.cfg" -mesh "mesh_RAE2822_turb_FFD.su2" -dv "DV_VALUE.npy" -np 1 -si 0 -cmin -10 -gflag "F" -mno 0.729 -reno 6500000 -aoa 2.31 
#python Bayesian_Shape_Optimization.py

#python -m examples.Multi_Seed_Wrapper --ntrain 40 --iters 100 --q 4 --func "DTLZ1" --ref_point  -20 -20 -20 --dim 4 --nobj 3 --ncons 0 --num_test 5 --start_seed 1023 --use_mtgp --use_partial --thresh 0.2

#mpirun -np 5

#srun python -m examples.Parallel_TC --ntrain 20 --iters 100 --q 5 --func "branincurrin" --ref_point -300 -20 --dim 2 --nobj 2 --ncons 0 --use_mtgp --use_partial --thresh 1e-6 --num_test 20 --start_seed 1023

#srun python -m examples.Parallel_TC --ntrain 40 --iters 20 --q 5 --func "DTLZ1" --ref_point -20 -20 -20 --dim 4 --nobj 3 --ncons 0 --use_mtgp --use_partial --thresh 1e-6 --num_test 5 --start_seed 1023

#srun python -m examples.All_Parallel_TC --ntrain 20 --iters 20 --q 5 --func "zdt3" --ref_point -1.1 -1.1 --dim 2 --nobj 2 --ncons 0 --use_mtgp --use_partial --thresh 1e-6 --num_test 5 --start_seed 1023

srun --mpi=pmi2 -n 10 python -m examples.All_Parallel_TC --ntrain 100 --iters 100 --q 4 --func "zdt3" --ref_point -11.0 -11.0 --dim 10 --nobj 2 --ncons 0 --acq "qpots" --num_test 10 --start_seed 1023 --use_mtgp --use_partial --thresh 1e-6


#python -m examples.1All_Parallel_TC_Testing_noMPI --ntrain 100 --iters 20 --q 4 --func "zdt3" --ref_point -11.0 -11.0 --dim 10 --nobj 2 --ncons 0 --acq "qpots" --num_test 1 --start_seed 1023 --use_mtgp --use_partial --thresh 1e-6


echo " "

echo "srun exited with return code $local_rc"

echo " "



# Job complete



echo "Job ending on `hostname` at `date`"
