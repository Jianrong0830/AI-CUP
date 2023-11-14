import pandas as pd
import numpy as np
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class DateTimeConvert:
    def categorize_date(date):
        return date // 10 + 1
        
    def categorize_time(time):
        return time // 21600 + 1
        
    def to_sec(time):
        time_str = str(time)
        time_str = time_str.zfill(6)
        h = int(time_str[:2])
        m = int(time_str[2:4])
        s = int(time_str[4:])
        s += h*60*60 + m*60
        return s

class Process:
    def clean():
        data_path='C:/dataset_1st/training.csv'
        data=pd.read_csv(data_path)
        
        # 狀態碼空值補0
        data['stscd'].fillna(0, inplace=True)
        
        # 刪除其他空值列
        data.dropna(inplace=True)

        # 轉換中文欄位
        columns_path='C:/dataset_1st/columns.csv'
        columns=pd.read_csv(columns_path)['訓練資料欄位中文說明']
        data.columns = columns
        
        # 刪除ID類欄位
        data.drop(['交易序號', '顧客ID', '交易卡號', '特店代號', '收單行代碼'], axis=1, inplace=True)

        data.to_csv('C:/dataset_1st/training_cleaned.csv', encoding='utf-8-sig', index=False)
        #print(data)
        
    def process1():
        data_path='C:/dataset_1st/training_cleaned.csv'
        data=pd.read_csv(data_path)
        
        # 授權時間轉換為秒數
        data['授權時間'] = data['授權時間'].apply(DateTimeConvert.to_sec)
        
        # 建立指示變數：分期期數_指示
        data['分期期數_指示'] = (data['分期期數'] == 0).astype(int)
        
        data.to_csv('C:/dataset_1st/training_process1.csv', encoding='utf-8-sig', index=False)
        #print(data)
        
    def process2():
        data_path='C:/dataset_1st/training_process1.csv'
        data=pd.read_csv(data_path)
        
        # 切分日期區段、時間區段
        data['日期區段'] = data['授權日期'].apply(DateTimeConvert.categorize_date)
        data['時間區段'] = data['授權時間'].apply(DateTimeConvert.categorize_time)
        
        data.to_csv('C:/dataset_1st/training_process2.csv', encoding='utf-8-sig', index=False)
        #print(data)
        
    def process3():
        data_path='C:/dataset_1st/training_process2.csv'
        data=pd.read_csv(data_path)
        
        all_features = set(data.columns.tolist()) - set(["盜刷與否"])
        log_transform_features = ["交易金額-台幣", "分期期數", "實付金額", "消費地金額"]
        quantitive_features = ["授權日期", "授權時間", "交易金額-台幣", "分期期數", "實付金額", "消費地金額"]
        remaining_features = list(all_features - set(quantitive_features))
        
        # 偏度或峰度過高變數：對數轉換
        for feature in log_transform_features:
            new_feature_name = f"{feature}(對數)"
            quantitive_features[quantitive_features.index(feature)] = new_feature_name
            data[new_feature_name] = np.log1p(data[feature])
            data.drop([feature], axis=1, inplace=True)
        
        # 資料標準化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[quantitive_features])
        dump(scaler, 'scalers/scaler1.joblib') 
        
        new_columns = []
        for feature in quantitive_features:
            new_columns.append(f"{feature}(標準化)")
        
        data_scaled_df = pd.DataFrame(data_scaled, columns=new_columns)
        data_remaining_df = data[remaining_features]
        
        temp = pd.concat([data_scaled_df,  data_remaining_df], axis=1)
        data = pd.concat([temp, data['盜刷與否']], axis=1)
        
        # 刪除空值列(再次確認)
        data.dropna(inplace=True)
        
        data.to_csv('C:/dataset_1st/training_process3.csv', encoding='utf-8-sig', index=False)
        #print(data)
        
    def preparatory():
        data_path='C:/dataset_1st/training_process3.csv'
        data=pd.read_csv(data_path)
        
        data.drop(['實付金額(對數)(標準化)', '消費地金額(對數)(標準化)', '是否分期交易'], axis=1, inplace=True)
        
        data.dropna(inplace=True)
        
        data.to_csv('C:/dataset_1st/training_preparatory.csv', encoding='utf-8-sig', index=False)
        #print(data)
        

class Analyze:
    def __init__(self, data, folder):
        self.data = data
        self.folder = folder
        
    def summary(self):
        print("摘要:")
        info_df = pd.DataFrame(columns=['特徵', '資料型態', '非空值數量'])
        for col in self.data.columns:
            dtype = self.data[col].dtype
            non_null_count = self.data[col].count()
            new_row = pd.DataFrame({'特徵': [col], '資料型態': [dtype], '非空值數量': [non_null_count]})
            info_df = pd.concat([info_df, new_row], ignore_index=True)
        
        info_df.to_csv('{}/摘要.csv'.format(self.folder), encoding="utf_8_sig")
        print(info_df, '\n')

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
        correlation_matrix_numerical.to_csv('{}/相關係數矩陣.csv'.format(self.folder), encoding="utf_8_sig")
        print(correlation_matrix_numerical, '\n')
    
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix_numerical, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("相關係數矩陣")
        plt.savefig("{}/相關係數矩陣.png".format(self.folder))
        plt.show()
