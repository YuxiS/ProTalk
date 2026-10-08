"""Load ProTalk YAML configuration without the unused Tacotron text package."""
from pathlib import Path
import yaml


def create_hparams(yaml_file=None):
    path = Path(yaml_file) if yaml_file else Path(__file__).with_name("hparams.yaml")
    with path.open(encoding="utf8") as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return HyperParams(params)


class HyperParams:
    def __init__(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, HyperParams(value) if isinstance(value, dict) else value)
