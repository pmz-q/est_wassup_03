import argparse
import json
from os.path import isfile, exists
from os import listdir, rename
from uuid import uuid4


def dfs(content:dict={"name": "root", "content": {}}, parent_path:str="../data"):
  for dir in listdir(parent_path):
    uuid = str(uuid4())
    if isfile(f"{parent_path}/{dir}"): # rename files
      if "." in dir:
        uuid = f"{uuid}.{dir.split('.')[-1]}"
      content["content"][uuid] = dir
      rename(f"{parent_path}/{dir}", f"{parent_path}/{uuid}")
    else: # rename dirs
      content["content"][uuid] = { "name": dir, "content": {} }
      new_path = f"{parent_path}/{uuid}"
      rename(f"{parent_path}/{dir}", new_path)
      dfs(content["content"][uuid], new_path)

def kor_to_uuid(cfg):
  root = cfg.dir_path
  
  filename_history = {
    "root": {
      "name": root,
      "content": {}
    }
  }
  
  if exists(root) and not isfile(root):
    dfs(filename_history["root"], root)
  
  with open(f"{root}/filename_mapper.json", "w") as f:
    json.dump(filename_history, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--dir-path", type=str, default="../data")
  
  config = parser.parse_args()
  kor_to_uuid(config)