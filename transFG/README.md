# transFG model for fine-grained visual classification 

<div align="center">

This model is from <a href="https://github.com/TACJu/TransFG">transFG</a> please check this repo which is open-source Pytorch implementation of the paper "TransFG: A Transformer Architecture for Fine-grained Recognition" (Ju He, Jie-Neng Chen, Shuai Liu, Adam Kortylewski, Cheng Yang, Yutong Bai, Changhu Wang, Alan Yuille).
</div>

<br>

## <div align="center">Documentation</div>


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
- prepare the Dataset. Dataset is downloaded <a href="https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71573">Here</a>

please check
- 
```bash

```
</details>


<details>

<summary>Inference</summary>

transFG inference with Pytorch.

```bash
python test.py
```

</details>

<details>
<summary>Train</summary>

run this command.

- google pre-trained ViT model. Model is downloaded <a href="https://console.cloud.google.com/storage/vit_models/?pli=1">Here</a> 

```bash

```
</details>