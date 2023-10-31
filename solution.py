import os
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

# убираем выбросы
df_sales = df_sales[
    df_sales['sales'] < 5000
    ]

# эксперимент проводился с 49 до 55 день включительно
df_sales_test = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]

# строим QUPED
df_sales_before = df_sales[
    df_sales['day'].isin(np.arange(28, 49))
]

# получим страты
bins = np.linspace(15, 65, 6).astype(int)
df_users['age_bin'] = pd.cut(df_users.age, bins=bins)
stratum = df_users.groupby(['gender', 'age_bin']).user_id.count() / len(df_users)

app = Flask(__name__)


@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test, True)
    return jsonify(has_effect=int(has_effect))


def _check_test(test, is_ratio=False):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    pvalue = calculate_p_value(df_sales_test, df_sales_before, group_a_one, group_a_two, is_ratio)
    if pvalue < 0.05:
        return False
    pvalue = calculate_p_value(df_sales_test, df_sales_before, user_a, user_b, is_ratio)

    return pvalue < 0.05


def calculate_p_value(df_test: pd.DataFrame, df_before: pd.DataFrame, user_a_list: List, user_b_list: List,
                      is_ratio: bool):
    if is_ratio:
        a, coef = calculate_linearization_metric(df_test, user_a_list, 'user_id', 'sales')
        a_before, coef_before = calculate_linearization_metric(df_before, user_a_list, 'user_id', 'sales')
        b, _ = calculate_linearization_metric(df_test, user_b_list, 'user_id', 'sales', coef)
        b_before, _ = calculate_linearization_metric(df_before, user_b_list, 'user_id', 'sales', coef_before)
    else:
        a = calculate_users_metric(df_test, user_a_list, 'user_id', 'sales')
        a_before = calculate_users_metric(df_before, user_a_list, 'user_id', 'sales')
        b = calculate_users_metric(df_test, user_b_list, 'user_id', 'sales')
        b_before = calculate_users_metric(df_before, user_b_list, 'user_id', 'sales')

    a = get_covariate_df(a, a_before, 'user_id', 'sales', 'sales_cov')
    b = get_covariate_df(b, b_before, 'user_id', 'sales', 'sales_cov')

    a['sales_cuped'], b['sales_cuped'] = calculate_quped_metric(a, b, 'sales', 'sales_cov')

    a_mean, a_var = calculate_stratified_metrics(a, 'user_id', 'sales_cuped', ['gender', 'age_bin'], stratum)
    b_mean, b_var = calculate_stratified_metrics(b, 'user_id', 'sales_cuped', ['gender', 'age_bin'], stratum)

    delta = a_mean - b_mean
    std = np.sqrt(a_var / len(a) + b_var / len(b))
    statistic = delta / std
    return (1 - norm.cdf(np.abs(statistic))) * 2


def calculate_theta(y_control, y_pilot, y_control_cov, y_pilot_cov) -> float:
    """Вычисляем Theta.

    y_control - значения метрики во время пилота на контрольной группе
    y_pilot - значения метрики во время пилота на пилотной группе
    y_control_cov - значения ковариант на контрольной группе
    y_pilot_cov - значения ковариант на пилотной группе
    """
    y = np.hstack([y_control, y_pilot])
    y_cov = np.hstack([y_control_cov, y_pilot_cov])
    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance
    return theta


def calculate_users_metric(df: pd.DataFrame, user_list: List, user_column: str, metric_name: str) -> pd.DataFrame:
    sales = df[
        df[user_column].isin(user_list)
    ].copy()
    result = sales.groupby(user_column, as_index=False)[metric_name].sum()
    return result


def get_covariate_df(df, df_before, user_column: str, metric_name, renamed_metric_name: str) \
        -> pd.DataFrame:
    df_before.rename(columns={metric_name: renamed_metric_name}, inplace=True)
    df = df.merge(df_before, how='left', on=user_column)
    return df


def calculate_quped_metric(a: pd.DataFrame, b: pd.DataFrame, metric_name: str, cov_metric_name: str) \
        -> Tuple[np.array, np.array]:
    y_control = a[metric_name].values
    y_control_cov = a[cov_metric_name].fillna(0).values
    y_pilot = b[metric_name].values
    y_pilot_cov = b[cov_metric_name].fillna(0).values

    theta = calculate_theta(y_control, y_pilot, y_control_cov, y_pilot_cov)
    a_cuped = y_control - theta * y_control_cov
    b_cuped = y_pilot - theta * y_pilot_cov
    return a_cuped, b_cuped


def calculate_stratified_metrics(df: pd.DataFrame, user_column: str, metric_name: str, stratified_columns: List,
                                 stratum: pd.Series):
    df = df.merge(df_users, how='left', on=user_column)
    avg = (df.groupby(stratified_columns)[metric_name].mean() * stratum).sum()
    var = (df.groupby(stratified_columns)[metric_name].var() * stratum).sum()
    return avg, var


def calculate_linearization_metric(df: pd.DataFrame, user_list: List, user_column: str, metric_name: str,
                                   coef: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
    sales = df[
        df[user_column].isin(user_list)
    ].copy()
    x = sales.groupby(user_column)[metric_name].sum()
    y = sales.groupby(user_column)[metric_name].count()
    if coef is None:
        coef = np.sum(x) / np.sum(y)
    lin = x - coef * y
    lin = lin.reset_index()
    lin.columns = [user_column, metric_name]
    return lin, coef

