import dill
import os
import glob
import logging
import json
from datetime import datetime

import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')

def predict() -> None:
    list_of_files = glob.glob(f'{path}/data/models/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    filename = f'{latest_file}'
    with open(filename, 'rb') as file:
        model = dill.load(file)
    lst = []
    for filename in os.listdir(f"{path}/data/test"):
        with open(os.path.join(f"{path}/data/test", filename), 'r') as f:
            text = json.loads(f.read())
            df = pd.DataFrame.from_dict([text])
            y = model.predict(df)
            lst.append(y)
    out = pd.DataFrame(lst, columns=['result'])
    print(out)
    # out.to_csv('data/predictions/pred.csv')
    out.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()