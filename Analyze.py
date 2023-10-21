import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Step 1: 資料讀取與預處理

    # 讀取資料
data_path='dataset_1st/training.csv'
data=pd.read_csv(data_path)

    # 空值處理
data['stscd'].fillna(0, inplace=True)   #狀態碼補0
data.dropna(inplace=True)               #刪除剩下空值列
data.to_csv('dataset_1st/training.csv')

    # 中文columns
columns_path='dataset_1st/columns.csv'
columns=pd.read_csv(columns_path)['訓練資料欄位中文說明']
data.columns = columns  # 更改欄位名稱

    # 輸出摘要
print("摘要:")
print(data.info(), '\n')

# Step 2: 單變數分析

    # 敘述統計
print("敘述統計:")
describe=data.describe()
describe.to_csv('analysis/敘述統計.csv', encoding="utf_8_sig")
print(describe, '\n')

    # 盜刷分布圖
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='盜刷與否', data=data)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.title('label')
plt.savefig("analysis/盜刷分布圖.png")
plt.show()

# Step 3: 雙變數分析(相關係數矩陣)

    # CSV
print("相關係數矩陣:")
numerical_features = data.select_dtypes(include=['number'])
correlation_matrix_numerical = numerical_features.corr()
correlation_with_label_numerical = correlation_matrix_numerical["盜刷與否"].sort_values(ascending=False)
correlation_matrix_numerical.to_csv('analysis/相關係數矩陣.csv', encoding="utf_8_sig")
print(correlation_matrix_numerical, '\n')

    # PNG
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_numerical, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("相關係數矩陣")
plt.savefig("analysis/相關係數矩陣.png")
plt.show()


# Step 4: 多變數分析
#（根據需求添加，例如使用 PCA 進行降維）

# Step 5: 額外的資料探索
#（根據需求添加，例如地理信息可視化）

# Step 6: 總結與報告
#（將您的觀察和結論寫入一個報告或 Jupyter Notebook）

