### モジュールのインポート
import pandas as pd

# bool型をint型に変換する
def bool_to_int(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df