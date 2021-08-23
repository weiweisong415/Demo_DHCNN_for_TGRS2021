---
DHCNN implementation via Pytorch
---
### 1. Running example:
Environment: python 3

Requirements:
```python
pytorch
torchvision
```
### 2. Statement:
In our paper, we adopt VGG-F as backbone network. However, pytorch doesn't provide pretrained VGG-F model, we use pretrained Alexnet to replace VGG-F. Thus, it is normal
to obtain the smaller-value maps obained by this code than the results reported in paper.

### 3. Data processing:
Download the UC Merced dataset from https://pan.baidu.com/s/1bEsOdaklaQgEKbFpq0mK1g , the passwords are: ucmd. 
### 4. Demo:
```python
python DHCNN_UCMD21.py
```
