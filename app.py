import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import get_data_for_prediction,PredictPipline


def prediction_datapoint():

    get_predict_object = get_data_for_prediction()
    pred_df = get_predict_object.get_data_as_dataframe()
    print(pred_df)

    predict_pipeline = PredictPipline()
    result = predict_pipeline.predict(pred_df)
    return result

if __name__ == '__main__':
    results = prediction_datapoint()
    print(results)