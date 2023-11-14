import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import warnings
import gc
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

# 讀取檔案
print('讀取檔案')
data_path='C:/dataset_1st/public_processed.csv'
data=pd.read_csv(data_path)

original_data = data.copy()

# 狀態碼空值補0
print('狀態碼空值補0')
data['stscd'].fillna(0, inplace=True)

# 轉換中文欄位
print('轉換中文欄位')
columns_path='C:/dataset_1st/columns.csv'
columns=pd.read_csv(columns_path)['訓練資料欄位中文說明']
columns.drop(25, inplace=True)
data.columns = columns

# 刪除ID類欄位
print('刪除ID類欄位')
data.drop(['交易序號', '顧客ID', '交易卡號', '特店代號', '收單行代碼'], axis=1, inplace=True)

# 授權時間轉換為秒數
print('授權時間轉換為秒數')
data['授權時間'] = data['授權時間'].apply(DateTimeConvert.to_sec)

# 建立指示變數：分期期數_指示
print('建立指示變數：分期期數_指示')
data['分期期數_指示'] = (data['分期期數'] == 0).astype(int)

# 切分日期區段、時間區段
print('切分日期區段、時間區段')
data['日期區段'] = data['授權日期'].apply(DateTimeConvert.categorize_date)
data['時間區段'] = data['授權時間'].apply(DateTimeConvert.categorize_time)

# 對數轉換 & 標準化
all_features = set(data.columns.tolist())
log_transform_features = ["交易金額-台幣", "分期期數", "實付金額", "消費地金額"]
quantitive_features = ["授權日期", "授權時間", "交易金額-台幣", "分期期數", "實付金額", "消費地金額"]
remaining_features = list(all_features - set(quantitive_features))

    # 對數轉換
print('對數轉換')
for feature in log_transform_features:
    new_feature_name = f"{feature}(對數)"
    quantitive_features[quantitive_features.index(feature)] = new_feature_name
    data[new_feature_name] = np.log1p(data[feature])
    data.drop([feature], axis=1, inplace=True)
    
    # 標準化
print('標準化')
scaler = load('scalers/scaler1.joblib')
data_scaled = scaler.fit_transform(data[quantitive_features])
new_columns = []
for feature in quantitive_features:
    new_columns.append(f"{feature}(標準化)")

data_scaled_df = pd.DataFrame(data_scaled, columns=new_columns)
data_remaining_df = data[remaining_features]

data = pd.concat([data_scaled_df,  data_remaining_df], axis=1)

# 刪除多重共線變數
print('刪除多重共線變數')
data.drop(['實付金額(對數)(標準化)', '消費地金額(對數)(標準化)', '是否分期交易'], axis=1, inplace=True)

#data.to_csv('C:/dataset_1st/public_processed_preparatory.csv', encoding='utf-8-sig', index=False)
#print(data)

# 變數分類
quantitive_features = ["授權日期(標準化)", "授權時間(標準化)", "交易金額-台幣(對數)(標準化)", "分期期數(對數)(標準化)"]
categorical_features = ["日期區段", "消費地幣別", "mcc_code", "交易類別", "交易型態", "支付型態", "消費城市", "時間區段", "消費地國別"]
booling_features = ["是否紅利交易", "分期期數_指示", "網路交易註記", "3D交易註記", "超額註記碼", "Fallback註記", "狀態碼"]

quantitive_features += booling_features

quantitive_data = data[quantitive_features]
categorical_data = [data[feature] for feature in categorical_features]

# 加載模型
print('加載模型')
model = load_model('models/tensorflow_model_test.h5')

# 預測
print('預測')
predictions = model.predict([quantitive_data] + categorical_data)
# 將 sigmoid 函數的輸出轉化為二元標籤
predicted_labels = (predictions > 0.5).astype(int)
predicted_labels_df = pd.DataFrame(predicted_labels, columns=['pred'])
submit = pd.concat([original_data['txkey'], predicted_labels_df], axis=1)

# 存檔
print('存檔')
submit.to_csv('submit.csv', index=False)
print(submit)