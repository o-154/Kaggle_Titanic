### モジュールのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

from sklearn.model_selection import train_test_split
# from . import general_processes

### ランダムフォレスト
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
    # 検証用データで予測した結果のclassfication_reportを表示する
    print(classification_report(y_val,y_val_pred))
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