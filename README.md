# LipNet

This is a Keras implementation of [LipNet: End-to-End Sentence-level Lipreading](https://paperswithcode.com/paper/lipnet-end-to-end-sentence-level-lipreading).

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/sttaseen/LipNet/assets/67076071/17a67afa-69ba-4d23-b9e6-94820adb8263"
  width=300
  height=auto
  ><br>
    <p style="font-size:1.5vw;">♪ ♫ ♪</p>
  </div>
</div>

## Requirements
### Setting up a conda environment

#### Install MiniConda
The following instructions are for Linux. For other operating systems, download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
```
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
 "Miniconda3.sh"
```
Install the .sh file.
```
bash Miniconda3.sh
```
Remove the installer:
```
rm Miniconda3.sh
```
#### Creating a virtual environment
Run the following commands to create a virtual environment and activate it:
```
conda create -n lips python=3.8 -y
conda activate lips
```
Make sure to run ```conda activate lips``` before running any of the scripts in this repo.

### Installing Dependencies

#### TensorFlow
The current implementation is in TensorFlow (since the original paper also used Tensorflow). There are plans to shift it to PyTorch in the future. 

Install Tensorflow using their official [step-by-step instructions](https://www.tensorflow.org/install/pip#:~:text=Step%2Dby%2Dstep-,instructions,-Linux).

#### Dependencies
Install other dependencies using:
```
pip install -r requirements.txt
```

This one is optional but to use the conda environment in Notebook, run:
```
conda install ipykernel -y
ipython kernel install --user --name=lips
```

## Setup
### Downloading and extracting the dataset

Extract [LRS2_v1](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) in the data folder such that the directory looks as below:

```
.
├── app
├── data
│   ├── lrs2_v1
│   │   └── mvlrs_v1
│   │       ├── main
│   │       │   ├── 5535415699068794046
│   │       │   └── ...
│   │       └── pretrain
│   │       │   ├── 5535415699068794046
│   │       │   └── ...
│   │       ├── pretrain.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       └── val.txt
│   └── ...
...
```
The ```./data/mvlrs_v1/main``` and ```./data/mvlrs_v1/pretrain```folders contain folders with videos and ground truths, and the ```.txt``` files contain the individual video numbers for each split.

### Training the model

All modelling parts are in the notebook ```LipNet.ipynb```.

### Running the model

For Streamlit, you need to have [FFmpeg](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) installed. Run the following in the terminal to run the model in Streamlit:

```
cd ./app
streamlit run ./streamlitapp.py
```


This implementation uses [CTC Loss](https://distill.pub/2017/ctc/).
