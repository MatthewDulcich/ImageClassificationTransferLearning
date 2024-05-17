# Image Classification with Transfer Learning

This project uses the CUB_200_2011 fine grained birds dataset to demonstrate the efficiency of transfer learning.

## Dataset

You can read about the dataset in the README provided in the dataset you download from [CUB_200_2011 Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)

## Setup

Make sure to have Python 3.11.9 installed. If you are on a Mac, start the venv with the command below, then skip to the Open file section:

```bash
source python-venv311/bin/activate
```

If you don't have a mac or the venv doesen't work you can make one with

```bash
python3.11 -m venv python-venv311
```

Then install the `requirements.txt`

```bash
pip install -r requirements.txt
pip install tensorflow
```
Download the data from [CUB_200_2011 Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) and put it in the main directory.

Open `image-classification.ipynb`
You can also open this file in Google Collaboratory (Make sure to install all the libraries in requirements.txt and change the data's directory path)

Run all the cells