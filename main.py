import pandas as pd
from functions import process_datasets

### 入出力データセットの名称
## 入力データ
train_file = 'train.csv'
test_file = 'test.csv'

### データのインポート
df_train = pd.read_csv('original_data/' + train_file)
df_test = pd.read_csv('original_data/' + test_file)
original_datasets = [df_train, df_test]

### データ加工の実施
model_input_datasets = []
model_input_datasets.append(process_datasets.pattern_1(*original_datasets))
model_input_datasets.append(process_datasets.pattern_2(*original_datasets))
model_input_datasets.append(process_datasets.pattern_3(*original_datasets))