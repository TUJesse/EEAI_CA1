import pandas as pd
from Config import *

def format_types_two(types: pd.Series) -> pd.Series:
    types = types.str.strip().str.lower().fillna(Config.EMPTY_TYPE)
    return types

# ref: https://www.w3schools.com/python/python_file_write.asp
def write_to_file(file, message):
    with open(file, "a") as f:
        f.write(message)