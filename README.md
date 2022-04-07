# Neural-Network-Classifier

## Install requirements
```
pip install -r requirements.txt
```

## Results and models
Test set acc: 98%.

Download the weights pretrained at [here](https://pan.baidu.com/s/1x7iWHLC_kEz-0ZqLOV9T-w)
Extraction code：e2yb

## Test
To test model on test set:
```
python FCnet.py --eval --resume-from /model_dir
```

## Training
To train model:
```
python FCnet.py --work-dir /save_model_dir
```
