import pandas as pd
import gc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gc.collect()

# SMOTE(分批)
print("SMOTE")
batch_size = 100000
smote = SMOTE()
X_resampled, y_resampled = [], []

for batch in range(0, X_train.shape[0], batch_size):
    X_batch = X_train.iloc[batch:batch + batch_size]
    y_batch = y_train.iloc[batch:batch + batch_size]
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
x = Dense(128, activation='relu')(quantitive_inputs)
gc.collect()

categorical_embedding_layers = []
categorical_inputs = []
for feature in categorical_features:
    vocab_size = int(X_train_resampled[feature].max() + 1)
    categorical_input = Input(shape=(1,), name=f'{feature}_input')
    categorical_inputs.append(categorical_input)
    categorical_embedding = Embedding(input_dim=vocab_size, output_dim=20, input_length=1)(categorical_input)
    categorical_embedding = Flatten()(categorical_embedding)
    categorical_embedding_layers.append(categorical_embedding)
    gc.collect()

merged_inputs = Concatenate()([x] + categorical_embedding_layers)
x = Dense(256, activation='relu')(merged_inputs)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
gc.collect()

# 模型編譯，加入其他評估指標
model = Model(inputs=[quantitive_inputs] + categorical_inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

# 輸出模型摘要
print("模型摘要：")
print(model.summary())
gc.collect()

# 訓練模型
print("訓練模型")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    [X_train_resampled[quantitive_features]] + [X_train_resampled[feature] for feature in categorical_features],
    y_train_resampled,
    epochs=10,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[early_stopping]
)
gc.collect()

# 評估模型
print("評估模型")
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    [X_test[quantitive_features]] + [X_test[feature] for feature in categorical_features],
    y_test
)
print(f"測試集上的損失為：{test_loss}")
print(f"測試集上的準確度為：{test_accuracy}")
print(f"測試集上的查準率為：{test_precision}")
print(f"測試集上的查全率為：{test_recall}")
print(f"測試集上的AUC值為：{test_auc}")

# 計算混淆矩陣
y_pred = model.predict([X_test[quantitive_features]] + [X_test[feature] for feature in categorical_features])
y_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("混淆矩陣：")
print(cm)

# 釋放記憶體
gc.collect()

# 儲存模型
print("儲存模型")
model_path = 'models/tensorflow_model_1.h5'
model.save(model_path)
gc.collect()
