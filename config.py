LOGGING_CONFIG = {
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'console': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }, 'file': {
        'class': 'logging.FileHandler',
        'formatter': 'default',
        'filename': './log.log'
    }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

MODEL_CONFIG = {
    'args': {
        'isTrain': 'false',
        'name': 'people',
        'Arc_path': './arcface_model/arcface_checkpoint.tar',
        'checkpoints_dir': './checkpoints'
    },
    'crop_size': 224,
    'insightface_path': './insightface_func/models'
}
