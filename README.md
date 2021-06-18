# Automated Audio Captioning (AAC)

Automated Audio Captioning training code on Clotho datasets with Listen-Attend-Spell and CNN-Tell models.

## TLDR for DCASE 2021 Task 6 challenge
```shell
git clone https://github.com/Labbeti/dcase2021task6
cd AAC
conda create -n env_aac python=3.9 pip
conda activate env_aac
pip install -e .
cd standalone
python download.py
cd ../slurm
./dcase.sh
```

## Installation
### Requirements
- Anaconda >= 4.8,
- java >= 1.8.0 for SPICE metric,
- Python dependencies can be installed with setup.py (if you use requirements.txt only you must run the shell script "post_setup.sh" manually).

### Environment
```shell
git clone https://github.com/Labbeti/dcase2021task6
cd AAC
conda create -n env_aac python=3.9 pip
conda activate env_aac
pip install -e .
```

This repository requires Java +1.8.0 and Stanford-CoreNLP for compute the 'Cider' and 'Spider' metrics.
On Ubuntu, Java can be installed with the following command :
```shell
sudo apt install default-jre
```

### Dataset and models installation
You can install the datasets with the script `standalone/download.py`. The default root path is `data`.
You can choose a dataset with the option `data=DATASET`.
This script also install language models for NLTK, spaCy and LanguageTool for process captions and a pre-trained model "Wavegram" from PANN.

Example : (download Clotho v2.1)
```shell
python download.py data=clotho
```

## Usage

### DCASE2021 Task 6
After install the environment and the dataset, juste run the script `dcase.sh` :
```shell
cd slurm
./dcase.sh
```

### Other example
Just run in directory `standalone` :
```shell
python train.py expt=lat data=clotho epochs=60 
```
For training Listen-Attend-Tell model with Clotho dataset during 60 epochs.
The testing is automatically done at the end of the training, but it can be turn off with `test=false`.

### Main options for `train.py`
This project use Hydra for parsing parameters in terminal. The syntax is `param_name=VALUE` instead of `--param_name VALUE`.

- expt=EXPERIMENT
	- lat (ListenAttendTell, a recurrent model based on Listen Attend Spell by Thomas Pellegrini)
	- cnnt (CNN-Tell, a convolutional recurrent model with a pre-trained encoder and the same decoder than LAT)

### Result directory
The model and result data are saved in `logs/Clotho/train_ListenAttendTell/{DATETIME}_{TAG}/` directory, where DATETIME is the date of the start of the process and TAG the value of the `tag` option.

The results directory contains :
- a `hydra` directory which store hydra parameters,
- a `checkpoint` directory which store the best and the last model among training,
- a ̀`events.outs.tfevents.ID` file which contains tensorboard logs,
- a `hparams.yaml` file which store the experiment model hyper-parameters,
- a `metrics.yaml` file which store the metrics results done by the Evaluator callback,
- a list of `result_SUBSET.csv` files for each test dataset SUBSET which store the output of the model for each sample.
- a `vocabulary.json` file containing the list of ordered words used, with frequencies of each word in the training dataset(s).

## External authors
- Thomas Pellegrini for the Listen-Attend-Spell model
	- [source code](https://github.com/topel/listen-attend-tell)
- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley for the Cnn14_DecisionLevelAtt model from PANN 
	- [source code](https://github.com/qiuqiangkong/audioset_tagging_cnn)
	- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
