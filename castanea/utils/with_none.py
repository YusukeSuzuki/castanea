
class WithNone:
    def __enter__(self): pass
    def __exit__(self,t,v,tb): pass

def device_or_none(x):
    return WithNone() if x is None else tf.device(x)

