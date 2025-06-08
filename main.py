import pandas as pd
from functions import specific_processes
from functions import models

response_variable_name = 'Survived'

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
model_input_datasets.append(specific_processes.pattern_1(*original_datasets))
model_input_datasets.append(specific_processes.pattern_2(*original_datasets))
model_input_datasets.append(specific_processes.pattern_3(*original_datasets))

### 予測の実行
accuracy_list = []
for i in range(len(model_input_datasets)):
    i = 2 # for test
    accuracy = models.newral_network(
        model_input_datasets[i][0],
        model_input_datasets[i][1],
        model_input_datasets[i][2],
        model_input_datasets[i][3],
        response_variable_name,
        i
    )
    accuracy_list.append(accuracy)
print(accuracy_list)