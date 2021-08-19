import logging
from abc import ABC, abstractmethod
from os import path
from typing import List

import cv2
import insightface
import numpy as np
import torch.nn.functional as F
from flask_app.commons.util import (FaceNotDetectedError, cv2_to_pil,
                                    dict_to_args, image_to_base64, pil_to_cv2,
                                    to_tensor)
from insightface_func.face_detect_crop_single import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from PIL import Image
from torchvision import transforms


class BaseModel(ABC):
    def __init__(self):
        self.predictor = None
        self.config = dict()

    @abstractmethod
    def init_model(self, config: dict):
        pass

    @abstractmethod
    def predict(self, inputs: dict):
        pass

    @abstractmethod
    def format_prediction(self, prediction):
        pass

    @abstractmethod
    def get_visualization(self, inputs, outputs):
        pass


class SimSwapModel(BaseModel):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self):
        super().__init__()
        self.app = None

    def init_model(self, config: dict):
        self.config = config
        args = dict_to_args(config['args'])
        opt = TestOptions().parse(args=args)
        self.predictor = create_model(opt)
        self.predictor.eval()
        self.app = Face_detect_crop(name=config['insightface']['model_name'],
                                    root=config['insightface']['model_dir'])
        self.app.prepare(ctx_id=config['insightface']['ctx_id'],
                         det_thresh=config['insightface']['detection_threshold'],
                         det_size=config['insightface']['det_size'])

    def predict(self, inputs: dict) -> Image.Image:
        """Adapted from test_wholeimage_swapsingle.py
        """
        crop_size = self.config['crop_size']

        # process source image
        src_img = inputs.get('src_img')
        src_img = pil_to_cv2(src_img)

        app_results = self.app.get(src_img, crop_size)
        if app_results == None:
            raise FaceNotDetectedError(img_type='source')
        src_img_list, _ = app_results

        src_img = self.transformer_Arcface(cv2_to_pil(src_img_list[0]))
        src_img_id = src_img.view(-1, src_img.shape[0], src_img.shape[1],
                                  src_img.shape[2])
        src_img_id = src_img_id.cuda()

        # create latent id
        src_img_id_downsample = F.interpolate(src_img_id, scale_factor=0.5)
        src_latend_id = self.predictor.netArc(src_img_id_downsample)
        src_latend_id = F.normalize(src_latend_id, p=2, dim=1)

        # process reference image
        ref_img = inputs.get('ref_img')
        ref_img = pil_to_cv2(ref_img)

        app_results = self.app.get(ref_img, crop_size)
        if app_results == None:
            raise FaceNotDetectedError(img_type='reference')
        ref_img_list, ref_mat_list = app_results

        # forward pass
        swap_result_list = []
        for i in ref_img_list:

            i_tensor = to_tensor(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))[None, ...]
            i_tensor = i_tensor.cuda()

            swap_result = self.predictor(None, i_tensor, src_latend_id,
                                         None, True)[0]
            swap_result_list.append(swap_result)

        output = self.reverse2wholeimage(swap_result_list,
                                         ref_mat_list, crop_size, ref_img)
        return output

    def format_prediction(self, prediction: Image.Image) -> str:
        return image_to_base64(prediction)

    def get_visualization(self, inputs, outputs):
        pass

    def reverse2wholeimage(self, swaped_imgs, mats, crop_size, oriimg) -> Image.Image:
        target_image_list = []
        img_mask_list = []
        for swaped_img, mat in zip(swaped_imgs, mats):
            swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
            img_white = np.full((crop_size, crop_size), 255, dtype=float)

            # inverse the Affine transformation matrix
            mat_rev = np.zeros([2, 3])
            div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
            mat_rev[0][0] = mat[1][1]/div1
            mat_rev[0][1] = -mat[0][1]/div1
            mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
            div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
            mat_rev[1][0] = mat[1][0]/div2
            mat_rev[1][1] = -mat[0][0]/div2
            mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

            orisize = (oriimg.shape[1], oriimg.shape[0])
            target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
            img_white = cv2.warpAffine(img_white, mat_rev, orisize)

            img_white[img_white > 20] = 255

            img_mask = img_white

            kernel = np.ones((10, 10), np.uint8)
            img_mask = cv2.erode(img_mask, kernel, iterations=1)

            img_mask /= 255

            img_mask = np.reshape(
                img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            target_image = np.array(
                target_image, dtype=np.float)[..., ::-1] * 255

            img_mask_list.append(img_mask)
            target_image_list.append(target_image)
        # target_image /= 255
        # target_image = 0
        img = np.array(oriimg, dtype=np.float)
        for img_mask, target_image in zip(img_mask_list, target_image_list):
            img = img_mask * target_image + (1-img_mask) * img

        final_img = img.astype(np.uint8)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_img)


class FaceDetectionModel(BaseModel):
    def __init__(self):
        super().__init__()

    def init_model(self, config: dict):
        self.config = config
        insightface_config = config['insightface']
        model_dir = insightface_config.get('model_dir')
        model_name = insightface_config.get('model_name')
        model_path = insightface_config.get('model_path')
        self.predictor = insightface.model_zoo.get_model(path.join(model_dir, model_name,
                                                                   model_path))
        assert self.predictor.taskname == 'detection', 'Taskname of insightface detector is not detection'
        self.predictor.prepare(insightface_config.get('ctx_id'),
                               input_size=insightface_config.get('det_size'))

    def predict(self, inputs: dict) -> List[float]:
        """
        Parameters
        ----------
        inputs : dict
            Dictionary with model inputs. Expects key `img` with PIL image for model.

        Returns
        -------
        output : List[float]
            Model output in [x1, y1, x2, y2, detection_score] format.
        """
        img = inputs.get('img')
        img = pil_to_cv2(img)
        detection_threshold = self.config['insightface']['detection_threshold']
        output, _ = self.predictor.detect(img, threshold=detection_threshold)
        output = output.flatten()
        return output

    def format_prediction(self, prediction: List[float]) -> dict:
        """
        Parameters
        ----------
        prediction : List[float]
            Model output in [`x1`, `y1`, `x2`, `y2`, `detection_score`] format.

        Returns
        -------
        formatted_output : dict
            Formatted output in dict format with keys: `is_face_detected`, `bounding_box`, `score`
            `bounding_box` is a dictionary with keys `x1`, `y1`, `x2`, `y2`
        """
        results = {}
        results['is_face_detected'] = len(prediction) != 0
        if results['is_face_detected']:
            bbox = {}
            bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'] = [
                int(i) for i in prediction[:-1]]
            results['bounding_box'] = bbox
            results['score'] = float(prediction[-1])

        return results

    def get_visualization(self, inputs, outputs):
        raise NotImplementedError
