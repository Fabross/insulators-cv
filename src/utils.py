import pandas as pd
from .pre_processing import Pre_processing
import os

def load_data(data_dir = '../InsulatorsDataSet/'):
    img_path = []
    img_data = []
    labels = []
    # extraction_features in format list( (extraction, kaolin(g/l)) )
    extraction_features = [('03','16'),('04','8'),('05','20'),('06','10'),('07', '6'),('08','25')]
    for i in range(len(extraction_features)):
        idx = extraction_features[i][0]
        kaolin = extraction_features[i][1]
        print(f"Loading {data_dir+idx+' Extraction/'} images...")
        for insulators_dir in os.listdir(data_dir+idx+' Extraction'):
            for img in os.listdir(data_dir+idx+' Extraction/'+insulators_dir):
                if img.split("_")[1] == "35.jpg":
                    pre_processing = Pre_processing(data_dir+idx+' Extraction/'+insulators_dir+"/"+img)
                    img_data += pre_processing.operate()
                    img_path += [data_dir+idx+' Extraction/'+insulators_dir+"/"+img]*5
                    labels += [kaolin]*5
    return pd.DataFrame({'img': img_path, 'preprocessed': img_data , 'label':labels})