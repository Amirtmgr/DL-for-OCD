# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

from enum import Enum

class _Datasets(Enum):
    """
    Keywords for the datasets' variables of config.yaml file.
    """

    PATH = "PATH"
    EVAL = "EVAL_PATH"
    BLACKLIST = "BLACK_LIST"
    WHITELIST = "WHITE_LIST"

class Config:
    """
    Baseclass to store keywords for different sections of config.yaml file.
    """
    DATASETS = _Datasets
    