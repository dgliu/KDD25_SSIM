## KDD25_SSIM

Experiments codes for the paper:

Dugang Liu, Chaohua Yang, Yuwen Fu, Xing Tang, Gongfu Li, Fuyuan Lyu, Xiuqiang He, Zhong Ming. Scenario Shared Instance Modeling for Click-through Rate Prediction. In Proceedings of SIGKDD '25.

**Please cite our SIGKDD '25 paper if you use our codes. Thanks!**


## Requirement

See the contents of requirements.txt


## Data Preprocessing

Please download the original data ([AliMama](https://tianchi.aliyun.com/dataset/56) and [AliCCP](https://tianchi.aliyun.com/dataset/408)) and place them in the corresponding directory of data.

You can prepare the AliMama data in the following code.

```
# process origin data
python Alimama_preprocess.py

# datatransform
python Alimama2tf.py
```

## Usage

An example of running SSIM:

```
# For AliMama
python  SSIM4trainer.py --dataset ali-mama --model dnn
```

## 

If you have any issues or ideas, feel free to contact us ([dugang.ldg@gmail.com](mailto:dugang.ldg@gmail.com)).

