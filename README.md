# berlin_workshop_bias

## Settings
Install the requirements.txt file, e.g. `pip install -r requirements.txt`. (to update with each addition of detection/mitigation methods.)

## Script Usage
Use `python generation.py` as the main file to launch detection/mitigation methods on specific models and datasets. 
* Add `--models mlp` to use a 3-layer mlp, or `resnet18`, `resnet34`, `resnet50`
* Add `--dataset_name cmnist` to use the ColoredMNIST dataset, alongside `--dataset_percent 0.5pct` between {0.5pct, 1pct, 2pct, 5pct}
* Add optional `--dataset_root path/to/root/before/the/dataset` 

* e.g; `python generation.py --dataset_name cmnist --dataset_root data/cmnist --dataset_percent 0.5pct`


## (Addition of Textual datasets :)
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

