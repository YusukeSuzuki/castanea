import tensorflow as tf

class LayerParameter:
    def __init__(self,
            padding='SAME', with_bias=False, with_weight_normalize=False, rectifier=None,
            var_device=None):
        self.padding = padding
        self.with_bias = with_bias
        self.with_weight_normalize = with_weight_normalize
        self.rectifier = rectifier
        self.var_device = var_device

