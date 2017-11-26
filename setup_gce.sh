# Make sure to scp cudnn-8.0-linux-x64-v6.0 tgz to the /tmp folder
sudo apt-get update
sudo apt-get upgrade
#Y
sudo apt-get install -y build-essential
cd /tmp
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Some user interaction here

source ~/.bashrc
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-8-0
cat <<EOF >> ~/.bashrc
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64
export PATH=\${CUDA_HOME}/bin:\${PATH}
EOF
source ~/.bashrc
cuda-install-samples-8.0.sh ~
cd ~/NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery
cd /tmp
tar xvzf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp -P cuda/include/cudnn.h $CUDA_HOME/include
sudo cp -P cuda/lib64/libcudnn* $CUDA_HOME/lib64
sudo chmod u+w $CUDA_HOME/include/cudnn.h
sudo chmod a+r $CUDA_HOME/lib64/libcudnn*

sudo apt-get install libcupti-dev
sudo apt-get install ffmpeg
conda install python=2.7.12
pip install tensorflow-gpu
pip install Pillow
pip install librosa
pip install soundfile
pip install tqdm


cd /home/sriramso/
git clone https://github.com/ASzot/voice-conversion.git
pip install virtualenv

pip install jupyter
jupyter notebook --generate-config
vim ~/.jupyter/jupyter_notebook_config.py
# add following lines
# c = get_config()
# c.NotebookApp.ip = '*'
# c.NotebookApp.open_browser = False
# c.NotebookApp.port = 8888



