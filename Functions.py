import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

class Categorize:
    def categorize_date(date):
        if date <= 13: return 1
        elif date > 13 and date <= 27: return 2
        elif date > 27 and date <= 41: return 3
        else: return 4
        
    def categorize_time(time):
        if time <= 110813: return 1
        elif time > 110813 and time <= 150940: return 2
        elif time > 150940 and time <= 185427: return 3
        else: return 4

class Process:
    def clean():
        data_path='C:/dataset_1st/training.csv'
        data=pd.read_csv(data_path)
        
        data['stscd'].fillna(0, inplace=True)
        data.dropna(inplace=True)

        columns_path='C:/dataset_1st/columns.csv'
        columns=pd.read_csv(columns_path)['訓練資料欄位中文說明']
        data.columns = columns

        data.to_csv('C:/dataset_1st/training_cleaned.csv', encoding='utf-8-sig', index=False)
        print(data)
        
    def process1():
        data_path='C:/dataset_1st/training_cleaned.csv'
        data=pd.read_csv(data_path)
        
        data['分期期數_指示'] = (data['分期期數'] == 0).astype(int)
        data['消費地國別_指示'] = (data['消費地國別'] == 0).astype(int)
        data['日期區段'] = data['授權日期'].apply(Categorize.categorize_date)
        data['時間區段'] = data['授權時間'].apply(Categorize.categorize_time)
        
        data.to_csv('C:/dataset_1st/training_p1.csv', encoding='utf-8-sig', index=False)
        print(data)
        
    def process2():
        data_path='C:/dataset_1st/training_p1.csv'
        data=pd.read_csv(data_path)
        
        data.drop('授權日期', axis=1, inplace=True)
        data.drop('授權時間', axis=1, inplace=True)
        data.drop('是否分期交易', axis=1, inplace=True)
        data.drop('分期期數', axis=1, inplace=True)
        data.drop('實付金額', axis=1, inplace=True)
        data.drop('消費地國別', axis=1, inplace=True)
        
        data.to_csv('C:/dataset_1st/training_p2.csv', encoding='utf-8-sig', index=False)
        print(data)
        
    def process3():
        data_path='C:/dataset_1st/training_p2.csv'
        data=pd.read_csv(data_path)
        
        interaction_features = [
            ['交易金額-台幣', '交易類別'],
            ['交易金額-台幣', '網路交易註記'],
            ['時間區段', '交易類別'],
            ['消費地國別_指示', '交易金額-台幣']]

        for feature_pair in interaction_features:
            interaction_term = f"{feature_pair[0]}_X_{feature_pair[1]}"
            data[interaction_term] = data[feature_pair[0]] * data[feature_pair[1]]
            
        poly_features = ['交易金額-台幣', '交易類別']
        for i in poly_features:
            data[i+'(平方)'] = data[i] * data[i]
        
        data.to_csv('C:/dataset_1st/training_p3.csv', encoding='utf-8-sig', index=False)
        print(data)
        
    def preparatory():
        data_path='C:/dataset_1st/training_p3.csv'
        data=pd.read_csv(data_path)
        print("讀取完成")
        
        # 偏度或峰度過高變數：對數轉換
        features = data.columns
        skewness = data.apply(lambda x: skew(x.dropna()), axis=0)
        kurtosis = data.apply(lambda x: kurtosis(x.dropna()), axis=0)
        
        high_skew_features = [feature for feature, value in zip(features, skewness) if np.abs(value) > 1]
        high_kurt_features = [feature for feature, value in zip(features, kurtosis) if np.abs(value) > 3]
        log_transform_features = list(set(high_skew_features + high_kurt_features))
        print("log_transform_features:", log_transform_features)
        
        for feature in log_transform_features:
            data[feature] = np.log1p(data[feature])
            
        # 常態分佈變數：標準化
        features = data.columns
        skewness = data.apply(lambda x: skew(x.dropna()), axis=0)
        kurtosis = data.apply(lambda x: kurtosis(x.dropna()), axis=0)
        
        normal_skew_features = [feature for feature, value in zip(features, skewness) if np.abs(value) < 0.5]
        normal_kurt_features = [feature for feature, value in zip(features, kurtosis) if np.abs(value - 3) < 1]
        normal_dist_features = list(set(normal_skew_features + normal_kurt_features))
        print("normal_dist_features: ", normal_dist_features)

        scaler = StandardScaler()
        data[normal_dist_features] = scaler.fit_transform(data[normal_dist_features])
        
        # 刪除空值列(再次確認)
        data.dropna(inplace=True)
        
        data.to_csv('C:/dataset_1st/training_preparatory.csv', encoding='utf-8-sig', index=False)
        print(data)
        

class Analyze:
    def __init__(self, data, folder):
        self.data = data
        self.folder = folder
        
    def summary(self):
        print("摘要:")
        file_name = "{}/摘要.txt".format(self.folder )
        file_mode = "w"
        file = open(file_name, file_mode)
        file.write(str(self.data.info()))
        file.close()
        print(self.data.info(), '\n')

    def describe(self):
        print("敘述統計:")
        describe=self.data.describe()
        describe.to_csv('{}/敘述統計.csv'.format(self.folder), encoding="utf_8_sig")
        print(describe, '\n')
    
    def label_distribution(self):
        print("盜刷分布")
        file_name = "{}/盜刷分布.txt".format(self.folder )
        file_mode = "w"
        file = open(file_name, file_mode)
        fraudulentCnt=sum(self.data['盜刷與否'])
        file.write("盜刷： {}, 非盜刷： {}".format(fraudulentCnt, len(self.data)-fraudulentCnt))
        file.close()
        print(self.data.info(), '\n')
    
    def correlation_matrix(self):
        print("相關係數矩陣:")
        numerical_features = self.data.select_dtypes(include=['number'])
        correlation_matrix_numerical = numerical_features.corr()
        #correlation_with_label_numerical = correlation_matrix_numerical["盜刷與否"].sort_values(ascending=False)
        correlation_matrix_numerical.to_csv('{}/相關係數矩陣.csv'.format(self.folder), encoding="utf_8_sig")
        print(correlation_matrix_numerical, '\n')
    
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix_numerical, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("相關係數矩陣")
        plt.savefig("{}/相關係數矩陣.png".format(self.folder))
        plt.show()
    