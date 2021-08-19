import logging
import time

from flask_app.model.declarations import FaceDetectionModel, SimSwapModel
from insightface_func.face_detect_crop_multi import Face

model_store = {}


def init_model_store(app):
    logging.info('Loading model store...')
    start_time = time.time()

    logging.info('Initializing SimSwapModel...')
    model_store['sim_swap'] = SimSwapModel()
    model_store['sim_swap'].init_model(app.config.get('MODEL_CONFIG'))

    logging.info('Initializing FaceDetectionModel...')
    model_store['face_detection'] = FaceDetectionModel()
    model_store['face_detection'].init_model(app.config.get('MODEL_CONFIG'))
    end_time = time.time()

    logging.info('Total time taken to load model store: %.3fs.' %
                 (end_time - start_time))
