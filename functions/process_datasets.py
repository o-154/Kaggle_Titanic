### モジュールのインポート
import pandas as pd
import numpy as np

def pattern_1(df_train, df_test):
    ### 学習データをxとyに分割
    train_y = df_train['Survived']
    train_x = df_train.drop(columns=['Survived'])

    ### テストデータのIDカラムのみ取得
    id_test = df_test['PassengerId']

    ### 学習データとテストデータの結合
    all_x = pd.concat([train_x, df_test])

    ### PassengerID
    ## 不要なので削除する
    all_x = all_x.drop(columns=['PassengerId'])
    
    ### Survived
    pass
    
    ### Name
    ## 1. 苗字を抽出する
    # 1.1 最初のカンマまでの文字列を抽出する関数を定義する
    def extract_before_comma(name):
        comma_index = name.find(",")
        return name[:comma_index].strip() if comma_index != -1 else None
    # 1.2 新しいカラム 'FamilyName' に抽出結果を入れる
    all_x['FamilyName'] = all_x['Name'].apply(extract_before_comma)
    ## 2. 同じ苗字の出現回数をカウントする
    all_x['SameFamilyName_Count'] = all_x.groupby('FamilyName')['FamilyName'].transform('count')
    ## 3. Name列とFamilyName列を削除する
    all_x = all_x.drop(['Name', 'FamilyName'], axis=1)
    
    ### Sex
    pass    
        
    ### Age
    pass

    ### SibSp
    pass

    ### Parch
    pass

    ### Ticket
    ## 1. 同じチケットの出現回数をカウントする
    all_x['SameTicket_Count'] = all_x.groupby('Ticket')['Ticket'].transform('count')
    ## 2. Ticket列を削除する
    all_x = all_x.drop(columns=['Ticket'])

    ### Fare
    pass
    
    ### Cabin
    ## 1. カラムの空白の有無をフラグ化
    all_x['Cabin_flag'] = all_x['Cabin'].apply(lambda x: 0 if pd.isna(x) or x == '' else 1)
    ## 2. Cabin列を削除する
    all_x = all_x.drop(columns=['Cabin'])

    ### Embarked
    all_x['Embarked'] = all_x['Embarked'].apply(lambda x: 'N' if pd.isna(x) or x == '' else x)

    ### カテゴリ変数化
    ## 対象：Sex, Cabin_flag, Embarked
    ## 方法：get_dummies関数
    all_x = pd.get_dummies(all_x, columns=['Sex', 'Cabin_flag', 'Embarked'])

    ### 学習データとテストデータに再分割（参考：Kaggleで勝つ P138）
    processed_train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
    processed_test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)
    processed_datasets = [processed_train_x, train_y, processed_test_x, id_test]
    return processed_datasets


def pattern_2(df_train, df_test):

    ### 学習データをxとyに分割
    train_y = df_train['Survived']
    train_x = df_train.drop(columns=['Survived'])

    ### テストデータのIDカラムのみ取得
    id_test = df_test['PassengerId']

    ### 学習データとテストデータの結合
    all_x = pd.concat([train_x, df_test])

    ### PassengerID
    ## 不要なので削除する
    all_x = all_x.drop(columns=['PassengerId'])

    ### Survived
    pass

    ### Pclass
    pass

    ### Name
    ### Titleを抽出し、6つにマージする
    ## 0. Titleの一覧を取得する
    # データの確認のために行う
    # titles = set()
    # for name in all_x['Name']:
    #     titles.add(name.split(',')[1].split('.')[0].strip())
    ## 1. NameとTitleの対応表を作る
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"
    }
    ## 2. Name→Titleの変換関数を定義する
    def get_titles():
        # 名前からTitleを取得する
        all_x['Title'] = all_x['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
        # 辞書を使って変換する
        all_x['Title'] = all_x.Title.map(Title_Dictionary)
        return all_x
    ## 3. 実際に変換する
    all_x = get_titles()
    ## 4. Name列を削除する
    all_x = all_x.drop(['Name'], axis=1)

    ### Sex
    pass

    ### Age
    ## 1. 3カラムをキーにグループ化し、各グループの年齢の中央値を求める
    grouped_train = all_x.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.Age.median()
    grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    def process_age(all_x, grouped_median_train):
        def fill_age(row):
            condition = (
                (grouped_median_train['Sex'] == row['Sex']) & 
                (grouped_median_train['Title'] == row['Title']) & 
                (grouped_median_train['Pclass'] == row['Pclass'])
            )
            return grouped_median_train[condition]['Age'].values[0]
        all_x['Age'] = all_x.apply(lambda row: fill_age(row) if pd.isna(row['Age']) else row['Age'], axis=1)
        return all_x
    # 呼び出し側（pattern_2の中）
    grouped_train = all_x.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.Age.median().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    all_x = process_age(all_x, grouped_median_train)

    ### SibSp
    pass

    ### Parch
    pass

    ### Ticket
    ## 1. 同じチケットの出現回数をカウントする
    all_x['SameTicket_Count'] = all_x.groupby('Ticket')['Ticket'].transform('count')
    ## 2. Ticket列を削除する
    all_x = all_x.drop(columns=['Ticket'])

    ### Fare
    pass

    ### Cabin
    ## 1. カラムの空白の有無をフラグ化
    all_x['Cabin_flag'] = all_x['Cabin'].apply(lambda x: 0 if pd.isna(x) or x == '' else 1)
    ## 2. Cabin列を削除する
    all_x = all_x.drop(columns=['Cabin'])

    ### Embarked
    ## 欠損値フラグ
    all_x['Embarked'] = all_x['Embarked'].apply(lambda x: 'N' if pd.isna(x) or x == '' else x)

    ### カテゴリ変数化
    ## 対象：Sex, Cabin_flag, Embarked
    ## 方法：get_dummies関数
    all_x = pd.get_dummies(all_x, columns=['Sex', 'Cabin_flag', 'Embarked', 'Title', 'Pclass'])

    ### 学習データとテストデータに再分割（参考：Kaggleで勝つ P138）
    processed_train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
    processed_test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)
    processed_datasets = [processed_train_x, train_y, processed_test_x, id_test]
    return processed_datasets


def pattern_3(df_train, df_test):
    ### 学習データをxとyに分割
    train_y = df_train['Survived']
    train_x = df_train.drop(columns=['Survived'])

    ### テストデータのIDカラムのみ取得
    id_test = df_test['PassengerId']

    ### 学習データとテストデータの結合
    all_x = pd.concat([train_x, df_test])

    ### PassengerID
    ## 不要なので削除する
    all_x = all_x.drop(columns=['PassengerId'])

    ### Survived
    pass

    ### Pclass
    pass

    ### Name
    ### Titleを抽出し、6つにマージする
    ## 0. Titleの一覧を取得する
    # データの確認のために行う
    titles = set()
    for name in all_x['Name']:
        titles.add(name.split(',')[1].split('.')[0].strip())
    ## 1. NameとTitleの対応表を作る
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"
    }
    ## 2. Name→Titleの変換関数を定義する
    def get_titles():
        # 名前からTitleを取得する
        all_x['Title'] = all_x['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
        # 辞書を使って変換する
        all_x['Title'] = all_x.Title.map(Title_Dictionary)
        return all_x
    ## 3. 実際に変換する
    all_x = get_titles()
    ## 4. Name列を削除する
    all_x = all_x.drop(['Name'], axis=1)

    ### Sex
    pass

    ### Age
    ## 1. 各行の3カラムの組み合わせに対応する年齢の中央値を返す関数を定義する
    def process_age(all_x, grouped_median_train):
        def fill_age(row):
            condition = (
                (grouped_median_train['Sex'] == row['Sex']) & 
                (grouped_median_train['Title'] == row['Title']) & 
                (grouped_median_train['Pclass'] == row['Pclass'])
            )
            return grouped_median_train[condition]['Age'].values[0]

        all_x['Age'] = all_x.apply(lambda row: fill_age(row) if pd.isna(row['Age']) else row['Age'], axis=1)
        return all_x
    # 2. 呼び出し側（
    grouped_train = all_x.iloc[:891].groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.Age.median().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
    all_x = process_age(all_x, grouped_median_train)

    ### SibSp
    ### Parch
    ## まとめてFamily_sizeカラムを作成する
    def process_family(all_x):
        all_x['FamilySize'] = all_x['SibSp'] + all_x['Parch'] + 1
        return all_x
    all_x = process_family(all_x)
    ##  SibSp,Parch列を削除する
    all_x = all_x.drop(['SibSp', 'Parch'], axis=1)

    ### Ticket
    ## 1. 同じチケットの出現回数をカウントする
    all_x['SameTicket_Count'] = all_x.groupby('Ticket')['Ticket'].transform('count')
    ## 2. Ticket列を削除する
    all_x = all_x.drop(columns=['Ticket'])

    ### Fare
    ## 全体の平均値で補間する
    def process_fares(all_x):
        all_x['Fare'] = all_x['Fare'].fillna(all_x.iloc[:891].Fare.mean())
        return all_x
    all_x = process_fares(all_x)

    ### Cabin
    ## 1文字目を採用する
    def process_cabin(all_x):
        # 1. 欠損値を'U'で補間する
        all_x['Cabin'] = all_x['Cabin'].fillna('U')
        # 2. カラムの内容を、1文字目に変換する
        all_x['Cabin'] = all_x['Cabin'].map(lambda c: c[0])
        return all_x
    all_x = process_cabin(all_x)

    ### Embarked
    ## 欠損値フラグ
    all_x['Embarked'] = all_x['Embarked'].apply(lambda x: 'N' if pd.isna(x) or x == '' else x)

    ### カテゴリ変数化
    ## 対象：Sex, Cabin_flag, Embarked
    ## 方法：get_dummies関数
    all_x = pd.get_dummies(all_x, columns=['Sex', 'Embarked', 'Title', 'Pclass', 'Cabin'])

    ### 学習データとテストデータに再分割（参考：Kaggleで勝つ P138）
    processed_train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
    processed_test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)
    processed_datasets = [processed_train_x, train_y, processed_test_x, id_test]
    return processed_datasets

### この関数は汎用関数にしたい
def bool_to_int(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df