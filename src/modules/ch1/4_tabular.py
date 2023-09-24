from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.core import URLs, untar_data, Normalize
from fastai.tabular.model import Categorify, FillMissing
from fastai.tabular.learner import tabular_learner, accuracy
from pathlib import Path

CATEGORY_NAMES = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
CONTINUOUS_NAMES = ['age', 'fnlwgt', 'education-num']

def get_path() -> Path:
  """
    Downloads and returns the ADULT_SAMPLE dataset path from fastai.
  """
  path =untar_data(URLs.ADULT_SAMPLE)
  return path

def train_model(path: Path) -> tabular_learner:
  dls= TabularDataLoaders.from_csv(
    path/'adult.csv',
    path=path,
    y_names="salary",
    cat_names=CATEGORY_NAMES,
    cont_names=CONTINUOUS_NAMES,
    procs=[Categorify, FillMissing, Normalize]
  ) 
  learn = tabular_learner(dls, metrics=accuracy)
  learn.fit_one_cycle(3)
  return learn

if __name__ == "__main__":
  path_to_dataset = get_path()
  trained_model = train_model(path_to_dataset)
  