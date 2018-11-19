from bittersweet.properties import get_chemopy_props_from_smilesfile
from bittersweet.model import Model
import pandas as pd
import argparse
import os

# User input
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Enter the path to input file")
parser.add_argument("dtype", help="Enter the molecular format of file (smiles/sdf/mol)")
parser.add_argument("output", help="Enter the path to the folder where output will be saved.")

# Model and model feature paths.
BITTER_MODEL = 'bittersweet/models/bitter_chemopy_rf_boruta.p'
BITTER_FEATURES = 'bittersweet/models/bitter_chemopy_boruta_features.p'
SWEET_MODEL = 'bittersweet/models/sweet_chemopy_rf_boruta.p'
SWEET_FEATURES = 'bittersweet/models/sweet_chemopy_boruta_features.p'

def get_prediction(f, dtype):
    
    
    if dtype == 'smiles':
        data = get_chemopy_props_from_smilesfile(f)
    else:
        raise Exception("{} is not a valid molecular format".format(dtype))
    

    # Convert to dataframe
    data = pd.DataFrame(data)

    # Find null rows
    null_values = data.isnull().sum(axis=1)
    null_rows = data.loc[null_values > 0]

    # Drop null rows
    data = data.loc[null_values == 0]

    # Generate prediction
    model = Model(BITTER_MODEL, BITTER_FEATURES,
                  SWEET_MODEL, SWEET_FEATURES)
    bitter_taste, bitter_prob, sweet_taste, sweet_prob = model.predict(data)

    data['bitter_taste'] = bitter_taste
    data['bitter_prob'] = bitter_prob[:, 1]

    data['sweet_taste'] = sweet_taste
    data['sweet_prob'] = sweet_prob[:, 1]

    return data, null_rows['name'].tolist()

if __name__ == "__main__":
    args = vars(parser.parse_args())
    data, not_predicted = get_prediction(args["input"], args["dtype"])
    data.to_csv(os.path.join(args["output"], 'output.csv'), encoding='utf-8')
    if len(not_predicted) != 0:
        with open(os.path.join(args["output"], 'not_predicted.txt'), 'w') as f:
            for item in not_predicted:
                f.write("{}\n".format(item))
        


