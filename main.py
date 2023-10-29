import pandas as pd
import Functions as f
import warnings
import gc
warnings.filterwarnings("ignore")

print("Preparatory")
f.Process.preparatory()
gc.collect()
data_path='C:/dataset_1st/training_preparatory.csv'
data=pd.read_csv(data_path)
Analysis=f.Analyze(data, "Analysis_preparatory")
Analysis.summary()
Analysis.describe()
Analysis.label_distribution()
Analysis.correlation_matrix()
gc.collect()
