import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image


def dict_to_args(dict: dict) -> list:
    args = []
    for key, value in dict.items():
        args.append(f'--{key}')
        args.append(f'{value}')
    return args


def pil_to_cv2(img: Image.Image) -> np.array:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.array) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def to_tensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def base64_to_image(base64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes))
    return image


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    image_bytes = base64.b64encode(buffered.getvalue())
    return image_bytes.decode("utf-8")
