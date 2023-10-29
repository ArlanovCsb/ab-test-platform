import os
import json
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm
from flask import Flask, jsonify, request

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

df_sales = df_sales[
    df_sales['sales'] < 5000
    ]

# эксперимент проводился с 49 до 55 день включительно
df_sales_test = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]

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
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    a = get_covariate_df(df_sales_test, df_sales_before, user_a, 'user_id', 'sales', 'sales_cov')
    b = get_covariate_df(df_sales_test, df_sales_before, user_b, 'user_id', 'sales', 'sales_cov')

    a['sales_cuped'], b['sales_cuped'] = calculate_quped_metric(a, b, 'sales', 'sales_cov')

    a_mean, a_var = calculate_stratified_metrics(a, 'user_id', 'sales_cuped', ['gender', 'age_bin'], stratum)
    b_mean, b_var = calculate_stratified_metrics(b, 'user_id', 'sales_cuped', ['gender', 'age_bin'], stratum)

    delta = a_mean - b_mean
    std = np.sqrt(a_var / len(a) + b_var / len(b))
    statistic = delta / std
    pvalue = (1 - norm.cdf(np.abs(statistic))) * 2
    return pvalue < 0.05


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


def calculate_users_metrics(df: pd.DataFrame, user_list: List, user_column: str, metric_name: str) -> pd.DataFrame:
    sales = df[
        df[user_column].isin(user_list)
    ]
    result = sales.groupby(user_column, as_index=False)[metric_name].sum()
    return result


def get_covariate_df(df, df_before, user_list: List, user_column: str, metric_name, renamed_metric_name: str) \
        -> pd.DataFrame:
    result = calculate_users_metrics(df, user_list, user_column, metric_name)
    cov = calculate_users_metrics(df_before, user_list, user_column, metric_name)
    cov.rename(columns={metric_name: renamed_metric_name}, inplace=True)
    result = result.merge(cov, how='left', on=user_column)
    return result


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


# def _check_test(test):
#     group_a_one = test['group_a_one']
#     group_a_two = test['group_a_two']
#     group_b = test['group_b']
#
#     user_a = group_a_one + group_a_two
#     user_b = group_b
#
#     sales_a = df_sales_test[
#         df_sales['user_id'].isin(user_a)
#     ]
#     a_x = sales_a.groupby('user_id').sales.sum()
#     a_y = sales_a.groupby('user_id').sales.count()
#     coef = np.sum(a_x) / np.sum(a_y)
#     a_lin = a_x - coef * a_y
#     sales_b = df_sales_test[
#         df_sales['user_id'].isin(user_b)
#     ]
#     b_x = sales_b.groupby('user_id').sales.sum()
#     b_y = sales_b.groupby('user_id').sales.count()
#     b_lin = b_x - coef * b_y
#
#     return ttest_ind(a_lin, b_lin)[1] < 0.05
