import config


def dataset(cls):
    for key, val in config.DATASET.items():
        setattr(cls, key, val)
    return cls
