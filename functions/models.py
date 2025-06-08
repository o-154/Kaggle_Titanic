### モジュールのインポート
import pandas as pd
import numpy as np

import os
import random

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# from . import general_processes


### 2. ランダムフォレスト
def random_forest(x_train_val, y_train_val, x_test, id_test, \
                  response_variable_name, features_number):
    model_name = 'random_forest'
    
    # モデルのインスタンスを作る
    model = RFC(random_state=0)

    # データを学習用、検証用に分割する(乱数シードを0、検証データの割合を0.3に指定)
    x_train,x_val,y_train,y_val = train_test_split(\
    x_train_val, y_train_val, test_size=0.3, random_state=0\
        )
    
    # fitメソッドで学習((説明変数,目的変数)を指定する)
    model.fit(x_train,y_train)

    ## 評価
    # 検証用データで予測値を生成する
    y_val_pred = model.predict(x_val)
    # # 検証用データで予測した結果のclassfication_reportを表示する
    # print(classification_report(y_val,y_val_pred))
    # # 混同行列を表示
    # cm = confusion_matrix(y_val,y_val_pred)
    # sns.heatmap(cm, annot=True, cmap='Blues')

    ##　解釈
    # モデルの特徴量の重要度を図示する
    # importances = model.feature_importances_
    # plt.figure(figsize=(10,10))
    # plt.barh(x_train_val.columns, importances)
    # plt.show()
    # accuracy（最後に返す）
    accuracy = accuracy_score(y_val,y_val_pred)

    ## 予測の実行
    y_test_pred = model.predict(x_test)
    df_result = pd.concat([id_test, \
                        pd.DataFrame(y_test_pred, columns=[response_variable_name])], \
                            axis=1)
    
    ## csvとして保存
    result_file_name = 'outputs/' + model_name + '_' + str(features_number+1) + '.csv'
    df_result.to_csv(result_file_name, index=False)

    return accuracy


### 3. ニューラルネットワーク
def newral_network(x_train_val, y_train_val, x_test, id_test, \
                  response_variable_name, features_number):
    model_name = 'newral_network'

    ## シード固定
    # シード固定関数の定義
    def set_seed(seed=42):
        random.seed(seed)  # Pythonの乱数
        np.random.seed(seed)  # NumPyの乱数
        tf.random.set_seed(seed)  # TensorFlowの乱数
        os.environ['PYTHONHASHSEED'] = str(seed)  # ハッシュのランダム性防止
    # シード固定の実行
    set_seed(42)

    ## データの前処理
    # 分割
    x_train,x_val,y_train,y_val = train_test_split(\
        x_train_val, y_train_val, test_size=0.3, random_state=0\
        )
    # 標準化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    ## モデルの構築
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train_val.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # バイナリ分類なのでsigmoid
    ])

    ## コンパイルと学習
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    ## 精度確認
    y_val_pred = (model.predict(x_val) > 0.5).astype(int)
    val_acc = accuracy_score(y_val, y_val_pred)
    # print(f"Validation Accuracy: {val_acc:.4f}")
    # # 混同行列を表示
    # from sklearn.metrics import confusion_matrix
    # import seaborn as sns
    # cm = confusion_matrix(y_val,y_val_pred)
    # sns.heatmap(cm, annot=True, cmap='Blues')

    # ## 解釈
    # # 特徴量計算関数を定義
    # def permutation_importance(model, x_val, y_val, metric=accuracy_score):
    #     baseline = metric(y_val, (model.predict(x_val) > 0.5).astype(int))
    #     importances = []

    #     for i in range(x_val.shape[1]):
    #         x_val_ = x_val.copy()
    #         np.random.shuffle(x_val_[:, i])  # i番目の特徴をシャッフル
    #         score = metric(y_val, (model.predict(x_val_) > 0.5).astype(int))
    #         importances.append(baseline - score)
        
    #     return np.array(importances)
    # # 関数の実行
    # importances = permutation_importance(model, x_val, y_val)
    # # 特徴量名とセットで表示
    # feature_names = x_train_val.columns.tolist()
    # for name, imp in zip(feature_names, importances):
    #     print(f"{name}: {imp:.4f}")

    ## 予測実行
    y_test_pred = (model.predict(x_test) > 0.5).astype(int)
    df_result = pd.concat([id_test, \
                        pd.DataFrame(y_test_pred, columns=[response_variable_name])], \
                            axis=1)
    
    ## 結果の出力
    result_file_name = 'outputs/' + model_name + '_' + str(features_number+1) + '.csv'
    df_result.to_csv(result_file_name, index=False)

    return val_acc