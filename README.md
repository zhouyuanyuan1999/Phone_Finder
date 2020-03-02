# Phone Finder

## Requirement
Python is installed.
Following packages are installed:
    pandas
    numpy
    matplotlib
    scikit-learn
    pytorch
    torchvision

## How to Train
    
There are many argument flags you can add. All the default settings are in utils/config.py file.
The model will be saved in model_dir (default is ./experiments/). In model_dir, there will be best model weights, train.log, and loss/acc plots. You can change model_dir using '--model_dir' flags.
Given the image folder is '../find_phone/',
To train from scratch, run:
```
python train_phone_finder.py --data_dir ../find_phone/
```
To continue previous training, run:
```
python train_phone_finder.py --data_dir ../find_phone/ --resume
```

Here is the list of command how the provided model (in experiments folder) is generated:
```
python train_phone_finder.py --data_dir ../find_phone/ --gpu 1 --rotation --pretrained --freezed_layers 60 --epochs 100 --model_dir ./experiments
python train_phone_finder.py --data_dir ../find_phone/ --gpu 1 --rotation --pretrained --freezed_layers 3 --epochs 300 -- model_dir ./experiments
```
Notice: --rotation flag make the images randomly rotate while training. Also the code will automatically split the dataset in 90%/10% train/validation split.
Notice: After running the above commands, the validation accuracy should be 100%, and train accuracy should be around 75-80%, however, without random rotation and translation, the accuracy on train dataset should be around 98%.
Notice: The default model is resnet18. 
Notice: During training, the validation accuracy will fluctuate. However, it is expected since the validation dataset has only around 10 images. 

## How to Predict
Pass the image path or path of the directory contains images as argument. Here are two exmaples:
To predict single image:
```
python find_phone.py ../find_phone/0.jpg
```
To predict multiple images (given all images are store in one directory):
```
python find_phone.py ../find_phone/
```
Notice: It is recommanded that you use the second command if you have multiple images, since with second command, you only need to load the model once which can save you some time. 

## Source

My code uses resnet [(arvix)](https://arxiv.org/pdf/1512.03385.pdf) to predict. 
I am using similar code structure of [my current lab project](https://github.com/hab-spc/hab-ml/tree/feature/instance)

## The directory structure
```
├── README.md          <- The top-level README for developers using this project.
├── data               <- Scripts to that load and transform images and labels
│   ├── dataloader.py
│   └── d_utils.py
│
├── models             <- Used to Create Models
│   └── model.py
├── utils         
│   ├── config.py      <- Script that contains default settings
│   ├── constants.py   <- Script that contains default constants
│   ├── eval_utils.py  <- Script that contains evaluation meters
│   └── logger.py      <- Script that creates logger
├── requirements.txt   <- The file that contains the names of required python packages
│
├── train_phone_finder.py   <- Script to train models 
│
├── train_phone_finder.py   <- Script to make predictions
│
└── trainer.py         <- Module to handle model training
```


