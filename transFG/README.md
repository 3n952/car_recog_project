# transFG model for fine-grained visual classification 

<div align="center">

This model is from <a href="https://github.com/TACJu/TransFG">transFG</a> please check this repo which is open-source Pytorch implementation of the paper "TransFG: A Transformer Architecture for Fine-grained Recognition" (Ju He, Jie-Neng Chen, Shuai Liu, Adam Kortylewski, Cheng Yang, Yutong Bai, Changhu Wang, Alan Yuille).
</div>

<br>

## <div align="center">how to use(usage)</div>


Clone repo

```bash
git clone https://github.com/3n952/car_recog_project.git  # clone
cd transFG
```

<details>
<summary>Dependencies</summary>

- python 3.7.3
- pytorch 1.5.1
- torchvision 0.6.1
- ml_collections
- apex

</details>

<details>
<summary>Data</summary>

run this command.

- prepare the google pre-trained ViT model. Model is downloaded <a href="https://console.cloud.google.com/storage/vit_models/?pli=1">Here</a> 

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```
- prepare the Dataset. Dataset is downloaded <a href="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71573">Here</a>.  
  please check this repo to figure out directory tree. Or modify modules in utils directory.  

- make custom dataset to train transfg model
```bash
cd preprocessing
python make_image_folder.py # crop the raw image data and save it as jpg file.
python text_to_df.py # split to train/val/test set and save as csv file.
```
</details>


<details>

<summary>Inference</summary>

transFG inference with Pytorch.

```bash
python test.py --batch_size 32
```

</details>

<details>
<summary>Train</summary>
transFG train with pytorch

For exmaple

```bash
python --m torch.distributed.launch --nproc_per_node=1 train.py --train_batch_size=32 --eval_batch_size=32 --split overlap --num_steps 100 --name my_transfg_run
```
</details>