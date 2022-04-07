# Neural-Network-Classifier

## Install requirements
```
pip install -r requirements.txt
```

## Results and models
test set acc: 98%.

download the weights pretrained at [here](https://drive.google.com/drive/folders/1hCaKNrlMF6ut0b36SedPRNC_434R8VVa?usp=sharing)

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
