import tkinter as tk
from model import BreastCancerDetectionModel
import pandas as pd

data = 'mypath/abc'

model = BreastCancerDetectionModel(data)

new_patient_data = pd.DataFrame()

diagnosis = model.predict(new_patient_data)

