import pathlib
import yaml

root = pathlib.Path(__file__).parent
for key, value in yaml.safe_load((root/'config.yaml').read_text()).items():
    globals()[key] = value