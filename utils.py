import plotly.express as px
import random
import numpy as np
import os
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """
    Устанавливает фиксированное значение начального состояния генераторов случайных чисел
    для обеспечения воспроизводимости экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_result(X_test: pd.DataFrame, y_test: pd.Series, preds: pd.Series) -> None:
    """
    Строит график сравнения реальных значений и предсказаний модели.
    """
    X_test['y_pred'] = preds
    X_test['y'] = y_test
    fig = px.line(X_test, x=[i for i in range(len(X_test))], y=['y', 'y_pred'],
                  labels={'value': 'Значение', 'ds': 'Дата'},
                  title='Реальные значения vs Предсказания')
    fig.update_layout(hovermode='x unified')
    fig.show()


def get_nan_intervals(df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Находит интервалы пропусков (NaN) в столбце Датафрейма.
    """
    mask = df['F1'].isna()
    changes = mask.astype(int).diff().fillna(0)
    starts = df.index[changes ==  1]
    ends  = df.index[changes == -1]

    if mask.iloc[0]:
        starts = starts.insert(0, df.index[0])
    if mask.iloc[-1]:
        ends = ends.append(pd.Index([df.index[-1]]))

    intervals = list(zip(starts, ends))
    return intervals


def reduce_mem_usage(df):
    """
    Проходит по всем столбцам DataFrame и изменяет тип данных
    для уменьшения потребления памяти.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
