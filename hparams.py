
import yaml
from text.symbols import symbols
def create_hparams(yaml_file='/remote-home/yfsong/code/prosody/StyleProsody/mellotron/hparams.yaml'):
    with open(yaml_file, mode='r', encoding='utf8') as f:
        hparams_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    hparams = HyperParams(hparams_dict)
    return hparams

class HyperParams:
    def __init__(self, params_dict):
        for key, value in params_dict.items():
            if isinstance(value, dict):
                self.__setattr__(key, HyperParams(value))
            else:
                self.__setattr__(key, value)
        self.__setattr__('n_symbols', len(symbols))
if __name__ == '__main__':
    params = create_hparams()
    
    print(params.pixmodel)