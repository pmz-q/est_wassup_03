from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent


def get_root_path() -> str:
  """
  Returns:
      str: "est_wassup_03"
  """
  return ROOT_PATH

