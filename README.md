# berlin_workshop_bias

## Textual datasets :
### Stereoset : 
Website : (website: https://stereoset.mit.edu/)

* Clone StereoSet github repository `git clone https://github.com/moinnadeem/StereoSet.git`

* Optional: Create and activate a virtual environment
`virtualenv stereo`
`source stereo/bin/activate`

* Comment `en-core-web-sm==2.2.5` in requirements.txt with a #

* Install required packages :
`pip install -U pip setuptools wheel
pip install spacy==2.2.4
python -m spacy download en_core_web_sm
cd StereoSet && pip install -r requirements.txt`

* Test if it works for a base model : 
`cd code`
`python3 eval_discriminative_models.py --pretrained-class bert-base-cased --tokenizer BertTokenizer --intrasentence-model BertLM --intersentence-model BertNextSentence --input-file ../data/dev.json --output-dir predictions/`

