# Multi-to-Single

This is the demo code for paper "Multi-to-Single: Reducing Multimodal Dependency in Emotion Recognition
through Contrastive Learning".

We have provided the structure of the model and the basic process of pretraining and fine-tuning. The code contains partial data of a subject (demo.npz for EEG modality and demo for EYE modality).

`data.py`: This file contains the data loading and processing code.

`CLUB.py, M2MCPC.py, models.py`: These files contain modules used in the M2S model.

`M2S.py`: This file contains the main architecture of the M2S model.

`pretrain.py`: This file contains the pretraining process.`

`finetune.py`: This file contains the fine-tuning process.

To pretrain the model, please run:
```
python3 pretrain.py
````

To finetune the model, please run:
```
python3 finetune.py --mode 'eeg2eye'
```
There are four modes in fine-tuning stage: eeg2eye, eye2eeg, eeg2eeg, eye2eye.
