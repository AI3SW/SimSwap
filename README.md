# AI3SW SimSwap

## SimSwap

* See [SimSwap README](SimSwap_README.md) for instructions on how to setup project i.e. download pretrained weights.
* [Preparation](docs/guidance/preparation.md)
* [Usage](docs/guidance/usage.md)

## Python Environment Management using Conda

```bash
$ # create conda environment
$ conda env create --file environment.yml
$ conda activate simswap
```

## Image generation using 1 source and 1 reference image

* Notebook demo can be found [here](notebooks/predict_function.ipynb).

## Face Detection using `insightface` library

* Notebook demo can be found [here](notebooks/face_detection.ipynb).

## Flask App

A Flask application can be setup that creates :

* a `predict` endpoint for SimSwap model that takes in the following parameters and returns a generated image encoded in base64 format:

    * a source image encoded in base64 format
    * a reference image encoded in base64 format

* a `detect` endpoint for face detection that takes in the following parameters and returns `is_face_detected`, `bounding_box`, and detection `score`:

    * a image encoded in base64 format

For a demo of the endpoints, refer to [test_flask](notebooks/test_flask.ipynb) notebook.

### Before Running

1. We use the face detection and alignment methods from **[insightface](https://github.com/deepinsight/insightface)** for image preprocessing. Please download the relative files and unzip them to ./insightface_func/models from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate).

1. There are two archive files in the drive: **checkpoints.zip** and **arcface_checkpoint.tar**

    - **Copy the arcface_checkpoint.tar into ./arcface_model**
    - **Unzip checkpoints.zip, place it in the root dir ./**

    [[Google Drive]](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R?usp=sharing)
    [[Baidu Drive]](https://pan.baidu.com/s/1wFV11RVZMHqd-ky4YpLdcA) Password: ```jd2v```

1. Update `MODEL_CONFIG` in [`config.py`](config.py) with path to pre-trained networks:

    * `Arc_path`
    * `checkpoints_dir`
    * `insightface.model_dir`

1. Create `instance` directory in project folder

1. Create ["deployment specific"](https://flask.palletsprojects.com/en/2.0.x/config/#instance-folders) config file `config.py` in `instance` directory with the following variables:
    * `SECRET_KEY`

### Using Flask Built-in Development Server

```bash
$ export FLASK_APP=flask_app
$ # export FLASK_ENV=development
$ flask run --host=0.0.0.0 --port=5000
```
