echo [$(date)]: "START"
echo [$(date)]: "creating conda env with python 3.8 version" 
conda create -n crop python==3.8 -y
echo [$(date)]: "activating the environment" 
conda activate crop
echo [$(date)]: "installing the requirements" 
pip install -r requirements.txt
echo [$(date)]: "END" 
