from enum import Enum

class Universe(Enum):
    SP500  = {"compo": "COMPO_SP500.parquet", "Price": "PX_LAST.parquet", "Price To Book":"PriceToBook.parquet", "ROE":"ROE.parquet"}