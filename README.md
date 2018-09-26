## A Mini Implementation Of CycleGAN


**Prerequisites**
1. Linux or OSX
2. Python 3.6+
3. Gluoncv
4. Mxnet


**Download  Dataset**
```bash
cd ./datasets/
python extract_data.py
```

**Run The Model**
```bash
python Cyclegan.py --num_epoch 50 --lr 0.00001
```

**Visualize The Data**

![sampleA](9_A.jpg "The Input A")

![sampleB](9_B.jpg "The Input B")


## References
["CycleGAN"](https://arxiv.org/abs/1703.10593)
