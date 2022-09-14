# berlin_workshop_bias

## Settings
Install the requirements.txt file, e.g. `pip install -r requirements.txt`. Note that this requirements.txt works at the very least for Python 3.7.9, maybe you need to allow more versions of pandas, numpy, ... if you use other python versions.

## Script Usage
Use `python generation.py` as the main file to launch detection/mitigation methods on specific models and datasets, when you are in your berlin_workshop_bias folder. 
* Add `--mitigation ldd` to use a mitigation method between 'ldd', 'lff', 'debian', or simply 'vanilla' to train a vanilla model.
* Add `--models MLP` to use a 3-layer mlp (or `resnet18`, `resnet34`, `resnet50` in the future.)
* Add `--dataset cmnist` to use the ColoredMNIST dataset,
* Alongside the CMNIST dataset, use `--percent 0.5pct` between {0.5pct, 1pct, 2pct, 5pct} to specify the percentage of conflict examples you want to use (not biased data = numbers of various colors in this case.)
* Add `--data_dir data` to specify the path to the data folder.

e.g : 
1) ldd : `python generation.py --mitigation ldd --model MLP --dataset cmnist --data_dir data --num_workers 4 --percent 5pct`
2) lff : `python generation.py --mitigation lff --model MLP --dataset cmnist --data_dir data --num_workers 4 --percent 5pct`
3) debian : `python generation.py --mitigation debian --model MLP --dataset cmnist --data_dir data --num_workers 4 --percent 5pct`
4) vanilla (based on ldd code) : `python generation.py --mitigation vanilla --model MLP --dataset cmnist --data_dir data --num_workers 4 --percent 5pct`



## (Addition of Textual datasets in the future :)
### Stereoset : 
Website : (website: https://stereoset.mit.edu/)

* Clone StereoSet github repository `git clone https://github.com/moinnadeem/StereoSet.git`

* Optional: Create and activate a virtual environment  
`virtualenv stereo`  
`source stereo/bin/activate`  

* Comment `en-core-web-sm==2.2.5` in requirements.txt with a #  

* Install required packages :  
`pip install -U pip setuptools wheel`  
`pip install spacy==2.2.4`  
`python -m spacy download en_core_web_sm`  
`cd StereoSet && pip install -r requirements.txt`  

* Test if it works for a base model :  
`cd code`  
`python3 eval_discriminative_models.py --pretrained-class bert-base-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file ../data/dev.json --output-dir predictions/`  

