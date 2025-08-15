Source codes for TMSTN.

# Setup

**Step. 1**: Create an environment includes the following packages: 

```
torch==1.11.0+cu113
torchvision==0.12.0+cu113
torchaudio==0.11.0+cu113
numpy==1.24.3
tqdm
h5py
scikit-image
scikit-learn
urllib3
matplotlib
opencv-python
seaborn
pillow
pandas
timm
scipy
randaugment
ml_collections
```

For ``conda``, you can create an environment using the following code (applicable to RTX4090).

```
conda create -n TMSTN python=3.9.12
conda activate TMSTN
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip3 install numpy==1.24.3 tqdm h5py scikit-image scikit-learn urllib3 matplotlib opencv-python seaborn pillow pandas timm scipy
# install randaugment
pip3 install git+https://github.com/ildoonet/pytorch-randaugment
# install ml-collections
pip3 install ml-collections
```

**Step 2**: Download the datasets, which can be found in [here](https://github.com/jindongwang/transferlearning/tree/master/data). Taking Office31 as an example, the structure of the dataset should be as follows:

```
Office31  
├─ images 
│  ├─ amazon
│  ├─ ...
│  └─ webcam
│     ├─ back_pack
│     		├─ frame_0001.jpg
│     		├─ frame_0002.jpg
│     		├─ ...
│     		└─ frame_0029.jpg
│     ├─ ...
│     └─ trash_can
```

**Step 3:** Download the Vision Transformer (ViT) pretrained model from [Google](https://github.com/jeonsworld/ViT-pytorch?tab=readme-ov-file). The models used in the experiments are [ViT-B_16](https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz) and [R50+ViT-B_16](https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/R50%2BViT-B_16.npz), respectively. Then, place the model in the following path:

```
../pretrained_models/imagenet21k+imagenet2012-FViT-B_16-224.npz
../pretrained_models/imagenet21k+imagenet2012_R50+ViT-B_16.npz
```

The path can be revised in the file ```models/loadViT.py```.

When steps 1-3 are completed, the environment is installed.

# Training

**Step 1**.  Train the source model using ```train_source.py``` as

```
python train_source.py --root_path your_dataset_path\Office31 --batch_size 32 --src amazon --tar dslr --num_classes 31 --use_cuda cuda --backbone vit --nepoch 30 --lr 0.008
```

​	When the experiment is complete, a pre-trained model will exist in ```./checkpoints/Office31/source/amazon```.

**Step 2**. With the source model, we can adapt the target model by ```train_target.py```, where the pre-trained model will be loaded from path ```./checkpoints/Office31/source/amazon/model_best.pkl```, and the values of hyperparameters of $\alpha$, $\beta$, $\gamma$, and $\delta$ can be found in the paper:

```python
python train_target.py --root_path your_dataset_path\Office31 --batch_size 32 --src amazon --tar dslr --num_classes 31 --use_cuda cuda --backbone vit --nepoch 30 --lr 0.008
```

With the above steps, the results of the paper could be reproduced.

# Acknowledgements

We would like to thank the following repositories for providing valuable insights: 

- [Transfer Learning Repo](https://github.com/jindongwang/transferlearning)

- [SFDA Repo](https://github.com/tntek/source-free-domain-adaptation)

- [CDTrans](https://github.com/CDTrans/CDTrans)



