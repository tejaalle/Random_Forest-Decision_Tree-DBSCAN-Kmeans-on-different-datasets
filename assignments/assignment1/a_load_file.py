from pathlib import Path
import pandas as pd


##############################################
# Implement the below method
# The method should be dataset-independent
##############################################
def read_dataset(path: Path) -> pd.DataFrame:
    # "utf-8" encoding is used to handle unicode in the csv file
    df = pd.read_csv(path,encoding="utf8")
    return df


if __name__ == "__main__":
    """
    In case you don't know, this if statement lets us only execute the following lines
    if and only if this file is the one being executed as the main script. Use this
    in the future to test your scripts before integrating them with other scripts.
    """
    dataset = read_dataset(Path('..', '..', 'iris.csv'))
    assert type(dataset) == pd.DataFrame
    print("ok")
