import pandas as pd
from .pre_processing import Pre_processing
import os
import numpy as np

def load_data(data_dir = '../InsulatorsDataSet/'):
    """Return dataset preprocessed

    Parameters
    ----------
    data_dir : str, optional
        data set directory, by default '../InsulatorsDataSet/'

    Returns
    -------
    array
        x_train, x_test, y_train, y_test
    """
    img_path = []
    img_data = []
    labels = []
    # extraction_features in format list( (extraction, kaolin(g/l)) )
    extraction_features = [('03',3),('04',1),('05',4),('06',2),('07', 0),('08',5)]
    for i in range(len(extraction_features)):
        idx = extraction_features[i][0]
        kaolin = extraction_features[i][1]
        print(f"Loading {data_dir+idx+' Extraction/'} images...")
        for insulators_dir in os.listdir(data_dir+idx+' Extraction'):
            for img in os.listdir(data_dir+idx+' Extraction/'+insulators_dir):
                if img.split("_")[1] == "35.jpg":
                    pre_processing = Pre_processing(data_dir+idx+' Extraction/'+insulators_dir+"/"+img)
                    pre_processing.operate()
                    for result in pre_processing.results:
                        img_data.append(result.reshape(-1))
                    img_path += [data_dir+idx+' Extraction/'+insulators_dir+"/"+img]*5
                    labels += [kaolin]*5
    return img_data, labels