# yamnet_model.py
import tensorflow as tf
from yamnet.yamnet import yamnet_frames_model
from yamnet.params import Params

def yamnet_model():
    # 1. Instancie la config par défaut
    params = Params()
    # 2. Crée l’architecture en lui donnant ces params
    model = yamnet_frames_model(params)
    # 3. Charge les poids HDF5
    model.load_weights("yamnet.h5")
    return model