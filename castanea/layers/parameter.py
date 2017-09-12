import tensorflow as tf

class BatchNormalizationParameter:
    def __init__(self, reuse=False, training=False):
        self.reuse = reuse
        self.training = training

class LayerParameter:
    def __init__(self,
            padding='SAME', with_bias=False, rectifier=None,
            with_weight_normalize=None, with_batch_normalize=None,
            var_device=None, var_scope_default_name=None, training=False):
        self.padding = padding
        self.with_bias = with_bias
        self.with_weight_normalize = with_weight_normalize
        self.with_batch_normalize = with_batch_normalize
        self.rectifier = rectifier
        self.var_device = var_device
        self.var_scope_default_name = var_scope_default_name
        self.training=training

