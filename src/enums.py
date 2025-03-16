from enum import Enum

class Universe(Enum):
    SP500  = {"compo": "COMPO_SP500.parquet", "Price": "PX_LAST.parquet", "Price To Book":"PTB.parquet", "ROE":"ROE.parquet"}

class FrequencyType(Enum):
    DAILY = 252 # 252 jours de trading dans une année
    WEEKLY = 52 # 52 semaines dans une année
    MONTHLY = 12 # 12 mois dans une année
    HALF_EXPOSURE = "HALF_EXPOSURE" # Exposition demie vie
    UNDESIRED_EXPOSURE = "UNDESIRED" # Exposition non désirée