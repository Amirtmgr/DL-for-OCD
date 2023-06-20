# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

from enum import Enum

# TODO(atm): Add more strings

class _WARNING(Enum):
    """WARNING Types
    """
    EMPTY = "File is empty."
    
class _ERROR(Enum):
    """ERROR Types
    """
    NO_FILE = "File Not Found."
    CRITICAL = "Critical Error"

class _INFO(Enum):
    INFORMATION = "INFORMATION"
    
class Verbose:
    """ Baseclass for verbose types.
    """
    
    WARNING = _WARNING
    ERROR = _ERROR
    INFO = _INFO