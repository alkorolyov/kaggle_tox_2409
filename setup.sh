CONDA_ENV="mlchem"

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
conda activate

conda run -n $CONDA_ENV python -m ipykernel install --user --name $CONDA_ENV
conda activate $CONDA_ENV