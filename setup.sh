CONDA_ENV="chem"

#curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
#bash Miniforge3-$(uname)-$(uname -m).sh -b
#source "${HOME}/miniforge3/etc/profile.d/conda.sh"
#source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
#conda activate
mamba env create
mamba activate
mamba run -n chem python -m ipykernel install --user --name chem