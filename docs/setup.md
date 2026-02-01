# Environment Setup
**1. Create Virtual Environment**
```
cd SparseOccVLA
python3.9 -m venv env
source env/bin/activate
```
**Conda** is also completely feasible, with only the difference being in the creation of the environment.

**2. Install Packages**
```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.37.1 
pip install mmcv-full==1.6.2
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
cd ..
pip install -r requirements.txt
```

**3. Install InternVL-Chat**
```
cd projects/mmdet3d_plugin/models/internvl_chat
pip install .
```

**4. Install language-packages** following [HERMES](https://github.com/LMD0311/HERMES/blob/main/docs/Usage.md)