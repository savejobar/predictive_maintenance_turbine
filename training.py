import copy
import optuna
import pandas as pd
import numpy as np
from optuna.trial import Trial
from lightgbm import LGBMRegressor
import lightgbm as lgb 

from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm


def config_fn(trial: Trial) -> dict[str]:
    """
    Формирует словарь гиперпараметров для модели LightGBM
    с использованием Optuna Trial.
    """
    boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'goss', 'dart'])
    config = {
        'boosting_type': boosting_type,
        'n_estimators': 2500,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 7, 127),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'random_state': 42,
    }

    if boosting_type == 'goss':
        top_rate = trial.suggest_float('top_rate', 0.2, 0.9)
        max_other = 1.0 - top_rate
        other_rate = trial.suggest_float('other_rate', 0.0, max_other)
        config['top_rate'] = top_rate
        config['other_rate'] = other_rate

    return config



def custom_objective(trial, train: pd.DataFrame, val: pd.DataFrame) -> float:
    """
    Кастомная функция для оптимизации гиперпараметров LightGBM через Optuna.
    """
    config = config_fn(trial)
    trial.set_user_attr("config", copy.deepcopy(config))

    X_train, y_train = train.drop(columns=['y']), train['y']
    X_val, y_val = val.drop(columns=['y']), val['y']

    model = LGBMRegressor(**config, verbose=-1)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, preds)
    print('Trial:', mape)
    return mape


def tuning_params(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """
    Подбор гиперпараметров для LGBMRegressor с помощью Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: custom_objective(trial, train, val),
        n_trials=75,
        show_progress_bar=True,
    )

    best_cfg = study.best_trial.user_attrs['config']
    return best_cfg


def tuning_pipeline(df: pd.DataFrame, val_size: int = 24 * 7) -> tuple:
    """
    Полный пайплайн подбора гиперпараметров и обучения модели LGBMRegressor.
    """
    train, val, test = df.iloc[:-2 * val_size], df.iloc[-2 * val_size:-val_size], df.iloc[-val_size:]

    best_model_config = tuning_params(train, val)
    model_config = best_model_config.copy()

    X_train, y_train = train.drop(columns=['y']), train['y']
    X_val, y_val = val.drop(columns=['y']), val['y']

    model = LGBMRegressor(**model_config, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
    preds = model.predict(pd.concat([val, test]).drop(columns=['y']))
    val_mape = mean_absolute_percentage_error(val['y'], preds[:val_size])
    test_mape = mean_absolute_percentage_error(test['y'], preds[val_size:])

    return preds, val_mape, test_mape, best_model_config, test


def get_importance_scores(
        model: LGBMRegressor, 
        X_val: pd.DataFrame, 
        y_val: np.ndarray, 
        n_repeats: int = 3
) -> dict:
    """
    Вычисляет важность признаков методом перестановок (permutation importance).
    """
    importance_scores = {}
    for col in tqdm(X_val.columns):
        scores = []
        for _ in range(n_repeats):
            X_permuted_val = X_val.copy()
            X_permuted_val[col] = np.random.permutation(X_permuted_val[col])

            preds = model.predict(X_permuted_val)

            val_mape = mean_absolute_percentage_error(y_val, preds)
            scores.append(val_mape)

        importance_scores[col] = np.mean(scores)
    return importance_scores


def make_permutation_and_feature_importance(
        df: pd.DataFrame, model_config: dict, val_size: int = 24 * 7):
    """
    Строит базовую и фильтрованную модель, вычисляет MAPE и важность признаков.
    """
    train = df.iloc[:-2 * val_size]
    val = df.iloc[-2 * val_size:-val_size]
    test  = df.iloc[-val_size:]

    X_train, y_train = train.drop(columns=['y']), train['y']
    X_val, y_val     = val.drop(columns=['y']), val['y']
    X_test, y_test   = test.drop(columns=['y']), test['y']

    model = LGBMRegressor(**model_config, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds = model.predict(pd.concat([X_val, X_test]))
    base_val_mape  = mean_absolute_percentage_error(y_val, preds[:val_size])
    base_test_mape = mean_absolute_percentage_error(y_test, preds[val_size:])

    print(f'Base MAPE: val={base_val_mape:.6f}, test={base_test_mape:.6f}')

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    )

    used_features = importance[importance > 0].index.tolist()

    print(f'Used features: {len(used_features)} / {X_train.shape[1]}')

    X_train_u = X_train[used_features]
    X_val_u   = X_val[used_features]
    X_test_u  = X_test[used_features]

    model_u = LGBMRegressor(**model_config, verbose=-1)
    model_u.fit(
        X_train_u, y_train,
        eval_set=[(X_val_u, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds_u = model_u.predict(pd.concat([X_val_u, X_test_u]))
    val_mape_u  = mean_absolute_percentage_error(y_val, preds_u[:val_size])
    test_mape_u = mean_absolute_percentage_error(y_test, preds_u[val_size:])

    print(f'Filtered MAPE: val={val_mape_u:.6f}, test={test_mape_u:.6f}')

    return {
        "base_model": model,
        "filtered_model": model_u,
        "base_mape": (base_val_mape, base_test_mape),
        "filtered_mape": (val_mape_u, test_mape_u),
        "features": used_features,
        "importance": importance.sort_values(ascending=False),
        "preds": preds_u
    }

