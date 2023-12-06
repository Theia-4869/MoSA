conda create -n mosa python=3.7
conda activate mosa

pip install -q tensorflow
# specifying tfds versions is important to reproduce our results
pip install tfds-nightly==4.4.0.dev202201080107
pip install opencv-python
pip install tensorflow-addons
pip install mock


conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install -c iopath iopath


# for transformers
pip install timm==0.4.12
pip install ml-collections

# Optional: for slurm jobs
pip install submitit -U
pip install slurm_gpustat