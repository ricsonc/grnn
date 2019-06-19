from munch import Munch

_options_ = {}

class OptionGroup(Munch):
    def __init__(self, name, *args, **kwargs):
        super().__init__()
            
        for arg in args:
            self.update(_options_[arg])
        for key in kwargs:
            self[key] = kwargs[key]
        if name:
            _options_[name] = self

            
def train_option(fn):
    return Munch(train_file = fn)

def test_option(fn):
    return Munch(test_file = fn)

def val_option(fn):
    return Munch(val_file = fn)

def data_options(name, fn, add_suffix = False):
    train_fn = fn+'_train' if add_suffix else fn
    test_fn = fn+'_test' if add_suffix else fn
    val_fn = fn+'_val' if add_suffix else fn
    
    return OptionGroup(name, train_file = train_fn, test_file = test_fn, val_file = val_fn)

def get(name):
    return _options_[name]
