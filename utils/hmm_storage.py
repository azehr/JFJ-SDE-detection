"""
Title: hmm_storage

Description:
    Saves and loads HMM model objects.
    Note: the root directory will need to be updated
    
"""

import pickle


def save_hmm_model(obj: object, filename: str):
    
    root = "C:\\Users\\abzeh\Documents\ETH Zurich\Masters Thesis\main\models\Saved HMMs"
    file = root + "\\" + filename 
    
    with open(file, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



def load_hmm_model(filename: str) -> object:
    
    root = "C:\\Users\\abzeh\Documents\ETH Zurich\Masters Thesis\main\models\Saved HMMs"
    file = root + "\\" + filename 
    
    with open(file, 'rb') as inp:
        model = pickle.load(inp)
    
    return model