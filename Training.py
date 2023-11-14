import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

param = {'noise_factor': 0.05, 'smote_batch_size': 100000,
         'dense_input': 128, 'dense':[128, 64, 32, 16],
         'learning_rate': 0.05, 'decay': 1e-6, 'cost_0': 67, 'cost_1': 1,
         'epochs': 15, 'train_batch_size': 2048}

# 變數分類
quantitive_features = ["授權日期(標準化)", "授權時間(標準化)", "交易金額-台幣(對數)(標準化)", "分期期數(對數)(標準化)"]
categorical_features = ["日期區段", "消費地幣別", "mcc_code", "交易類別", "交易型態", "支付型態", "消費城市", "時間區段", "消費地國別"]
booling_features = ["是否紅利交易", "分期期數_指示", "網路交易註記", "3D交易註記", "超額註記碼", "Fallback註記", "狀態碼"]
y_feature = "盜刷與否"

# 數值變數與二元變數合併
quantitive_features += booling_features

# 讀取數據
print("讀取數據")
data_path = 'C:/dataset_1st/training_preparatory.csv'
data = pd.read_csv(data_path)
gc.collect()

# 分割數據
print("分割數據")
X = data.drop(y_feature, axis=1)
y = data[y_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
gc.collect()


# 添加隨機噪聲
print("添加隨機噪聲")
X_train = X_train + param['noise_factor'] * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
gc.collect()


# SMOTE(分批)
print("SMOTE")
smote = SMOTE()
X_resampled, y_resampled = [], []

for batch in range(0, X_train.shape[0], param['smote_batch_size']):
    X_batch = X_train.iloc[batch:batch + param['smote_batch_size']]
    y_batch = y_train.iloc[batch:batch + param['smote_batch_size']]
    X_resampled_batch, y_resampled_batch = smote.fit_resample(X_batch, y_batch)
    X_resampled.append(X_resampled_batch)
    y_resampled.append(y_resampled_batch)
    gc.collect()

X_train_resampled = pd.concat(X_resampled)
y_train_resampled = pd.concat(y_resampled)
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled)
gc.collect()

# 定義模型
print("定義模型")
quantitive_inputs = Input(shape=(len(quantitive_features),), name='quantitive_input')
x = Dense(param['dense_input'], activation='relu')(quantitive_inputs)
gc.collect()

categorical_embedding_layers = []
categorical_inputs = []
for feature in categorical_features:
    vocab_size = int(X[feature].max() + 100)
    categorical_input = Input(shape=(1,), name=f'{feature}_input')
    categorical_inputs.append(categorical_input)
    categorical_embedding = Embedding(input_dim=vocab_size, output_dim=20, input_length=1)(categorical_input)
    categorical_embedding = Flatten()(categorical_embedding)
    categorical_embedding_layers.append(categorical_embedding)
    gc.collect()

merged_inputs = Concatenate()([x] + categorical_embedding_layers)
x = Dense(param['dense'][0], activation='relu')(merged_inputs)
x = Dropout(0.2)(x)
for i in param['dense']:
   x = Dense(i, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
   x = Dropout(0.2)(x) 
output = Dense(1, activation='sigmoid')(x)

gc.collect()

# 優化器、初始學習率、學習率衰減
optimizer = tf.keras.optimizers.Adam(learning_rate=param['learning_rate'], decay=param['decay'])

# 模型編譯，加入其他評估指標
model = Model(inputs=[quantitive_inputs] + categorical_inputs, outputs=output)
model.compile(optimizer=optimizer, loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# 輸出模型摘要
print("模型摘要：")
print(model.summary())
gc.collect()

# 訓練模型
print("訓練模型")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
class_weights = {0: param['cost_0'], 1: param['cost_1']}
history = model.fit(
    [X_train_resampled[quantitive_features]] + [X_train_resampled[feature] for feature in categorical_features],
    y_train_resampled,
    epochs=param['epochs'],
    batch_size=param['train_batch_size'],
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weights
)
gc.collect()

# 評估模型
print("評估模型")
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    [X_test[quantitive_features]] + [X_test[feature] for feature in categorical_features],
    y_test
)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f"Loss：{test_loss}")
print(f"Accuracy：{test_accuracy}")
print(f"Precision：{test_precision}")
print(f"Recall：{test_recall}")
print(f"AUC：{test_auc}")
print(f"F1：{test_f1}")

'''
# 計算混淆矩陣
y_pred = model.predict([X_test[quantitive_features]] + [X_test[feature] for feature in categorical_features])
y_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣：")
print(cm)
'''

# 釋放記憶體
gc.collect()

# 儲存模型
model_name = 'tensorflow_model_test'

print("儲存模型")
model_path = f"models/{model_name}.h5"
model.save(model_path)
gc.collect()

print("儲存評估結果")
with open(f"models/{model_name}_evaluation.txt", 'w', encoding='utf-8') as file:
    file.write(f"Loss：{test_loss}\n")
    file.write(f"Accuracy：{test_accuracy}\n")
    file.write(f"Precision：{test_precision}\n")
    file.write(f"Recall：{test_recall}\n")
    file.write(f"AUC：{test_auc}\n")
    file.write(f"F1：{test_f1}\n")
    #file.write(f"Confusion Matrix：{str(cm)}\n")
    file.write(f"param：\n{param}\n")
