from enum import Enum, IntEnum

# HandWashing Type
class HandWashingType(IntEnum):
    NoHandWash = 0
    Routine = 1
    Compulsive = 2

# CSV Header Type
class CSVHeader(Enum):
    """Enum class CSV headers
    
     0   timestamp    float64
     1   datetime     object 
     2   acc x        float64
     3   acc y        float64
     4   acc z        float64
     5   gyro x       float64
     6   gyro y       float64
     7   gyro z       float64
     8   user yes/no  float64
     9   compulsive   float64
     10  urge         float64
     11  tense        float64
     12  ignore       int64  
     13  relabeled    int64  
    """
    TIMESTAMP = "timestamp"
    DATETIME = "datetime"
    ACC_X = "acc x"
    ACC_Y = "acc y"
    ACC_Z = "acc z"
    GYRO_X = "gyro x"
    GYRO_Y = "gyro y"
    GYRO_Z = "gyro z"
    HW = "user yes/no"
    COMPULSIVE_HW = "compulsive"
    TENSE = "tense"
    URGE ="urge"
    IGNORE = "ignore"
    RELABELED = "relabeled"
    
# Metrics Type
class Metrics(Enum):
    Accuracy = "accuracy"
    Precision = "precision"
    Recall = "recall"
    F1 = "f1"
    Sample = "sample"
    MCC = "mcc"
    ROC = "roc"
    ROC_PR = "roc_pr"