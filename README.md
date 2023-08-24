## Noisy-ArcMix / STSASgram-MFN (Pytorch Implementation)
By Soonhyeon Choi (csh5956@kaist.ac.kr) at 
Korea Advanced Institute of Science and Technology (KAIST)

## Introduction
Noisy-ArcMix is a training technique that synthesizes anomalous samples by mixup and adds a correct or incorrect margin on the target angle to prevent overfitting on normal samples while maintaining intra-class compactness and increasing the angular difference between normal and abnormal features. Additionally, STSASgram architecture plays a critical role in extracting the spectrally attended log-Mel spectrogram (SASgram) through spectral attention as well as the temporal feature (Tgram) and log-Mel spectrogram (Sgram) for the purpose of anomalous sound detection.
<br/>
<br/>
This repository contains the implementation used for the results in our paper <Link>.

## TASTgram Architecture

<p align="center">
  <img src="./TASTgramMFN" alt="TASTgram">
</p>

The overall architecture of STSASgram. The temporal feature (Tgram) extracted from the raw wave through a CNN-based TgramNet is concatenated with the log-Mel spectrogram (Sgram) and the global spectral feature (SASgram) obtained from the log-Mel spectrogram through the spectral attention block.


## Noisy-ArcMix

<p align="center">
  <img src="./Angle_distribution.png" alt="STSASgram">
</p>
Histogram with the distribution of angles between samples and their corresponding centers for the Fan machine type in cases of (a) ArcFace, (b) ArcMix, and (c) Noisy-ArcMix.

## Datasets

[DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) Dataset: 
+ [development dataset](https://zenodo.org/record/3678171)
+ [additional training dataset](https://zenodo.org/record/3727685)
+ [Evaluation dataset](https://zenodo.org/record/3841772)


## Organization of the files

```shell
├── check_points/
├── datasets/
    ├── fan/
        ├── train/
        ├── test/
    ├── pump/
        ├── train/
        ├── test/
    ├── slider/
        ├── train/
        ├── test/
    ├── ToyCar/
        ├── train/
        ├── test/
    ├── ToyConveyor/
        ├── train/
        ├── test/
    ├── valve/
        ├── train/
        ├── test/
├── model/
├── Dockerfile
├── README.md
├── config.yaml
├── LICENSE
├── dataloader.py
├── eval.py
├── losses.py
├── train.py
├── trainer.py
├── utils.py
```

## Training
Check the `config.yaml` file to select a training mode from ['arcface', 'arcmix', 'noisy_arcmix']. Default is noisy-arcmix. 
<br/>
Use `python train.py` to train a model. 
```
python train.py
```

## Evaluating
Use `python eval.py` to evaluate the trained model.
```
python eval.py
```

## Model weights
Our trained model weights file can be accessed in https://drive.google.com/drive/folders/1tuUS-MKcAy-jFDVVdD5rpy-NU-3Pk46a?hl=ko.

## Experimental Results
 | machine Type | AUC(%) | pAUC(%) | mAUC(%) |
 | --------     | :-----:| :----:  | :----:  |
 | Fan          | 97.08  | 94.13   | 91.20   |
 | Pump         | 94.71  | 86.67   | 88.15   |
 | Slider       | 99.45  | 97.12   | 97.85   |
 | Valve        | 99.93  | 99.66   | 99.85   |
 | ToyCar       | 96.37  | 89.72   | 88.54   |
 | ToyConveyor  | 80.16  | 66.71   | 70.84   |
 | __Average__      | __94.74__  | __89.00__   | __89.41__   |




## Citation
If you use this method or this code in your paper, then pleas cite it:
