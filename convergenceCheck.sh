#!/bin/bash
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --partition=mi50
#SBATCH -o output
#SBATCH -e output

#EXAMPLES=("quadraticVelocities.py" "constantVelocity.py" "stommelGyre.py")
EXAMPLES=("isolatedDipoleVortex.py" "constantVelocity.py" "quadraticVelocities.py" "isolatedVortex.py" "stommelGyre.py")

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/joe/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/joe/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/joe/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/joe/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# --nx is the total number of xc points, including the edge masked values
# --ny is the total number of yc points, including the edge masked values
#
# --tolerance is the tolerance for the IRAM 
# [TO DO] --cgtolerance is the tolerance for the preconditioned conjugate gradient solver
# [TO DO] --pctolerance is the tolerance for the preconditioner
# [TO DO] --preconditioner is the preconditioner type ( Options : [jacobi,none] )
#
conda activate xnma

timestamp=$(date +"%Y-%m-%d-%H-%M")
odir="checks_${timestamp}"

mkdir -p ${odir}

prec="float64"
iramtol="1e-4"
cgtol="1e-12"
pctol="1e-2"
nx=(13 23 33 43)
# 53 63 73 83)
div=(4 2 1) # Set up for 25%, 50% and 100% eigenmode calculation
#div=(4)
for e in "${EXAMPLES[@]}"; do
  for d in "${div[@]}"; do
    for n in "${nx[@]}"; do

      # Calculate the maximum number of eigenmodes, based on the number of wet vorticity points
      m=$(( ((n-4)*(n-4) - 1)/d ))
    
      echo "$p $n $m"
      # run the example
      python ./examples/$e --nx $n --ny $n --nmodes $m --tolerance $iramtol --precision $prec
      
    done
  done
  mkdir -p $odir/$e
  mv *.png *.csv $odir/$e
done

