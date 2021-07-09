import logging
import time

from flask_app.model.declarations import SimSwap

model_store = {}


def init_model_store(app):
    logging.info('Loading model store...')
    start_time = time.time()

    model_store['sim_swap'] = SimSwap()
    model_store['sim_swap'].init_model(app.config.get('MODEL_CONFIG'))
    end_time = time.time()

    logging.info('Total time taken to load model store: %.3fs.' %
                 (end_time - start_time))
