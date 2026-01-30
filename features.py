import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures

def encode_dates(df: pd.DataFrame, column: str, season: int) -> pd.DataFrame:
    """
    Выполняет циклическое кодирование временного признака с использованием
    синусо-косинусного преобразования на заданный период.
    """
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / season)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / season)
    return df


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Функция формирует расширенный набор признаков для временного ряда, 
    включая временные, лаговые, оконные статистики, экспоненциальное 
    сглаживание, разности лагов и полиномиальные преобразования.
    """
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Временные признаки
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['hour'] = df['Date'].dt.hour

    df = encode_dates(df, 'month', 12)
    df = encode_dates(df, 'day_of_month', 30)
    df = encode_dates(df, 'day_of_week', 7)
    df = encode_dates(df, 'hour', 24)
    df.drop(columns=['month', 'day_of_month', 'day_of_week', 'hour'], inplace=True)

    # Лаги
    for lag in [*list(range(1, 25)), 72, 168]:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Скользящие окна
    window_sizes = [6, 12, 24, 72]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window).mean()
        df[f'rolling_median_{window}'] = df['y'].shift(1).rolling(window=window).median()
        df[f'rolling_std_{window}'] = df['y'].shift(1).rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['y'].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['y'].shift(1).rolling(window=window).max()
        df[f'rolling_sum_{window}'] = df['y'].shift(1).rolling(window=window).sum()

    # Expanding окна
    df[f'expandig_mean'] = df.y.shift(1).expanding().mean()
    df[f'expandig_median'] = df.y.shift(1).expanding().median()
    df[f'expandig_std'] = df.y.shift(1).expanding().std()
    df[f'expandig_min'] = df.y.shift(1).expanding().min()
    df[f'expandig_max'] = df.y.shift(1).expanding().max()
    df[f'expandig_sum'] = df.y.shift(1).expanding().sum()

    # EWM
    alphas = [0.2, 0.5, 0.8]
    for alpha in alphas:
        df[f'emv_alpha_"{alpha}"'] = df.lag_1.ewm(alpha=alpha).mean()

    # Дельты лагов
    for i in range(2, 15):
        df[f"delta_lag_{i}-lag_{i - 1}"] = df[f"lag_{i}"] - df[f"lag_{i - 1}"]

    # Полиномы
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_lags = 12
    poly_cols = [f"lag_{i}" for i in range(1, poly_lags + 1)] + [f"delta_lag_{i + 1}-lag_{i}" for i in range(1, poly_lags)]
    
    data_poly_input = df[poly_cols].fillna(0)
    data_poly = poly.fit_transform(data_poly_input)

    poly_feature_names = poly.get_feature_names_out(poly_cols)
    df_poly = pd.DataFrame.from_records(data_poly, columns=poly_feature_names, index=df.index)
    
    data_full = df.drop(columns=poly_cols).merge(df_poly, how='outer', left_index=True, right_index=True)

    data_full.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in data_full.columns]
    data_full = data_full.loc[:, ~data_full.columns.duplicated()]
    
    return data_full


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция подготавливает временной ряд к обучению модели, включая очистку
    данных, формирование вспомогательных признаков и генерацию признаков.
    """
    df = df[['Date', 'F1']].copy()
    df['Date'] = pd.to_datetime(df['Date'])

    df['is_run'] = (~df.F1.isna()).astype(int)
    df = df.fillna(0)

    hour_off = []
    hour_on = []
    cur_on = 1
    cur_off = 1
    for i in range(len(df)):
        if df.loc[i, 'is_run'] == 1:
            hour_off.append(0)
            hour_on.append(cur_on)
            cur_on += 1
            cur_off = 1
        else:
            hour_off.append(cur_off)
            cur_off += 1
            hour_on.append(0)
            cur_on = 1
    df['hour_off'] = hour_off
    df['hour_on'] = hour_on

    df = df.rename(columns={'F1': 'y'})

    df = make_features(df)

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    return df

