#!/bin/bash
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --partition=mi50
#SBATCH -o output
#SBATCH -e output

prog="eigenmodesSimpleTest.py"
prec="float64"
iramtol="1e-4"
cgtol="1e-12"
pctol="1e-2"
nx=(13 23 33 43)

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


conda activate xnma

timestamp=$(date +"%Y-%m-%d-%H-%M")
odir="checks_${timestamp}"

mkdir -p ${odir}


for n in "${nx[@]}"; do
    
    echo "$n"
    # run the example
    python ./examples/$prog --nx $n --ny $n --nmodes 80 --tolerance $iramtol --precision $prec


done
mkdir -p $odir/$prog
mv *.png *.csv $odir/$prog

