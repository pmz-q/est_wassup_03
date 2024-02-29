from pathlib import Path
from typing import Literal
import argparse


ROOT_PATH = Path(__file__).parent

CONFIG_PATH_MAPPER = {
  "main": f"{ROOT_PATH}/configs/config.py",
}

DESC_TITLE_MAPPER = {
  "main": "Let's do the facial expression classification!"
}

def get_args_parser(
  add_help=True,
  config_type: Literal["main"]="main"
):
  parser = argparse.ArgumentParser(description=DESC_TITLE_MAPPER[config_type], add_help=add_help)
  parser.add_argument("-c", "--config", default=CONFIG_PATH_MAPPER[config_type], type=str, help="configuration file")
  return parser
