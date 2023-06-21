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

From the root of the repository, run:
```
python get_data.py
```
This should download and extract the [GRID](https://paperswithcode.com/sota/lipreading-on-grid-corpus-mixed-speech) dataset into the ```./data``` folder.
```
.
├── app
├── data
│   ├── alignments
│   │   └── s1
│   │       ├── bbaf2n.align
│   │       └── ...
│   └── s1
│       ├── bbaf2n.mpg
│       └── ...
...
```
The ```./data/alignments/s1``` folder contains the ground-truth or alignments in the ```.align``` format, and the input videos are stored in the ```./data/s1``` folder.

### Training the model

All modelling parts are in the notebook ```LipNet.ipynb```.

### Running the model

For Streamlit, you need to have [FFmpeg](https://www.hostinger.com/tutorials/how-to-install-ffmpeg) installed. Run the following in terminal to run the model in streamlit:

```
streamlit run ./app/streamlitapp.py
```

TODO: The model can be run using Steamlit.

[CTC Loss](https://distill.pub/2017/ctc/)
