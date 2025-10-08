from importlib.resources import files
from .perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
PREPACKAGED_MODELS_DIR = files(__name__).joinpath("pretrained")
