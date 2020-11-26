import pandas as pd
import numpy as np
import lightgbm as lgb
from functools import reduce
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 300)

# b表上筛选出来的特征
b_col = ['id', 'x_num_0', 'x_num_1', 'x_num_2', 'x_num_3', 'x_cat_0',
             'x_cat_3', 'x_cat_4', 'x_cat_9', 'x_cat_10', 'x_cat_11']
# m表上筛选出来的特征
m_col = ['id', 'timestamp', 'x_num_70', 'x_num_5', 'x_num_6', 'x_num_7', 'x_num_10',
             'x_num_13', 'x_num_14', 'x_num_15', 'x_num_17', 'x_num_18', 'x_num_20',
             'x_num_22', 'x_num_23', 'x_num_24', 'x_num_25', 'x_num_26', 'x_num_27',
             'x_num_28', 'x_num_30', 'x_num_32', 'x_num_33',
             'x_num_39', 'x_num_41', 'x_num_43', 'x_num_45', 'x_num_46',
             'x_num_47', 'x_num_49', 'x_num_50', 'x_num_51', 'x_num_55',
             'x_num_63', 'x_num_68', 'x_num_69']
# m表上去掉id和timestamp后剩下的特征
more_cols = ['x_num_70', 'x_num_5', 'x_num_6', 'x_num_7', 'x_num_10',
             'x_num_13', 'x_num_14', 'x_num_15', 'x_num_17', 'x_num_18', 'x_num_20',
             'x_num_22', 'x_num_23', 'x_num_24', 'x_num_25', 'x_num_26', 'x_num_27',
             'x_num_28', 'x_num_30', 'x_num_32', 'x_num_33',
             'x_num_39', 'x_num_41', 'x_num_43', 'x_num_45', 'x_num_46',
             'x_num_47', 'x_num_49', 'x_num_50', 'x_num_51', 'x_num_55',
             'x_num_63', 'x_num_68', 'x_num_69']
# 阈值
threshold = 0.25

# 读取数据
def read_file(path):
    train_b = pd.read_csv(path + 'data_b_train.csv')
    valid_b = pd.read_csv(path + 'data_b_test.csv')
    train_m = pd.read_csv(path + 'data_m_train.csv')
    valid_m = pd.read_csv(path + 'data_m_test.csv')
    train_y = pd.read_csv(path + 'y_train.csv')

    return train_b, valid_b, train_m, valid_m, train_y

# 原始特征筛选
def original_feature_select(train_b, valid_b, train_m, valid_m):

    train_b = train_b[b_col]
    valid_b = valid_b[b_col]

    train_m = train_m[m_col]
    valid_m = valid_m[m_col]

    # 测试集缺失值处理
    valid_m.replace(-99, np.nan, inplace=True)
    valid_b.replace(-99, 0, inplace=True)

    return train_b, valid_b, train_m, valid_m

# 新建特征
class CreateFeature(object):
    def __init__(self):
        pass

    # 修改新增特征的名字
    def change_name(self, train, valid, suffix):
        names = train.columns.tolist()
        if 'id' in names:
            names.remove('id')

        names = [col + suffix for col in names]
        names = ['id'] + names

        train.columns = names
        valid.columns = names
        return train, valid

    # 在行为表上的特征衍生
    def pipeline(self, train, valid, more_cols):
        # 最后一天
        train_last = train.sort_values('timestamp', ascending=False).groupby('id', as_index=False).first()
        valid_last = valid.sort_values('timestamp', ascending=False).groupby('id', as_index=False).first()

        train_last.drop(columns='timestamp', inplace=True)
        valid_last.drop(columns='timestamp', inplace=True)

        train = train[['id'] + more_cols]
        valid = valid[['id'] + more_cols]

        # max
        train_max = train.groupby('id', as_index=False).max()
        valid_max = valid.groupby('id', as_index=False).max()

        # mean
        train_mean = train.groupby('id', as_index=False).mean()
        valid_mean = valid.groupby('id', as_index=False).mean()

        # std
        train_std = train.groupby('id', as_index=False).std()
        valid_std = valid.groupby('id', as_index=False).std()
        train_std['id'] = train_mean['id']
        valid_std['id'] = valid_mean['id']

        # cov
        train_cov = train_std / train_mean
        valid_cov = valid_std / valid_mean
        train_cov['id'] = train_mean['id']
        valid_cov['id'] = valid_mean['id']

        # median
        train_median = train.groupby('id', as_index=False).median()
        valid_median = valid.groupby('id', as_index=False).median()

        # mean / last
        train_mean_div_last = train_mean / train_last
        valid_mean_div_last = valid_mean / valid_last
        train_mean_div_last['id'] = train_mean['id']
        valid_mean_div_last['id'] = valid_mean['id']

        # max + last
        train_max_add_last = train_max + train_last
        valid_max_add_last = valid_max + valid_last
        train_max_add_last['id'] = train_mean['id']
        valid_max_add_last['id'] = valid_mean['id']

        # 修改衍生特征的列名
        train_last, valid_last = self.change_name(train_last, valid_last, '_last')
        train_max, valid_max = self.change_name(train_max, valid_max, '_max')
        train_mean, valid_mean = self.change_name(train_mean, valid_mean, '_mean')
        train_cov, valid_cov = self.change_name(train_cov, valid_cov, '_cov')
        train_std, valid_std = self.change_name(train_std, valid_std, '_std')
        train_median, valid_median = self.change_name(train_median, valid_median, '_median')
        train_max_add_last, valid_max_add_last = self.change_name(train_max_add_last, valid_max_add_last, '_max+last')
        train_mean_div_last, valid_mean_div_last = self.change_name(train_mean_div_last, valid_mean_div_last, '_mean/last')

        # 特征合并
        train_info = [train_last, train_cov, train_max, train_std, train_median, train_mean_div_last,
                      train_max_add_last]
        train_info = reduce(lambda left, right: pd.merge(left, right, on='id', how='left'), train_info)

        valid_info = [valid_last, valid_cov, valid_max, valid_std, valid_median, valid_mean_div_last,
                      valid_max_add_last]
        valid_info = reduce(lambda left, right: pd.merge(left, right, on='id', how='left'), valid_info)

        return train_info, valid_info

    # 基本信息表和行为表特征的衍生
    def pipeline2(self, train, valid):
        '''
        该部分的新建特征方式有：
        1. groupby(cat) * last_cols
        2. cat之间 ^
        3. x_num_0 * x_cat_9
        4. x_num_47_last * x_cat_0
        5. m表上的 num / num
        6. m表上的log(num)
        '''

        train_result = pd.DataFrame({'id': train['id']})
        valid_result = pd.DataFrame({'id': valid['id']})

        # 1.groupby(cat) * last_cols
        last_cols = [col + '_last' for col in more_cols]
        cat_cols = ['x_cat_0', 'x_cat_3', 'x_cat_4', 'x_cat_9', 'x_cat_10', 'x_cat_11']

        cat_col = cat_cols[0]
        train_df = train[[cat_col] + last_cols]
        valid_df = valid[[cat_col] + last_cols]

        # [x_cat_0, last_cols]
        for col in last_cols:
            train_result[col + '_' + cat_col + '_*median'] = train_df[col] * train_df.groupby(cat_col)[col].transform(
                np.median)
            valid_result[col + '_' + cat_col + '_*median'] = valid_df[col] * valid_df.groupby(cat_col)[col].transform(
                np.median)

        # [x_cat_9, last_cols]
        cat_col = cat_cols[3]
        train_df = train[[cat_col] + last_cols]
        valid_df = valid[[cat_col] + last_cols]
        for col in last_cols:
            train_result[col + '_' + cat_col + '_-median'] = train_df[col] - train_df.groupby(cat_col)[col].transform(
                np.median)
            valid_result[col + '_' + cat_col + '_-median'] = valid_df[col] - valid_df.groupby(cat_col)[col].transform(
                np.median)

        # 2. cat之间 ^
        cat_cols = ['x_cat_0', 'x_cat_10', 'x_cat_11']
        for i in range(len(cat_cols) - 1):
            for j in range(i + 1, len(cat_cols)):
                train_result[cat_cols[i] + '^' + cat_cols[j]] = train[cat_cols[i]] ^ train[cat_cols[j]]
                valid_result[cat_cols[i] + '^' + cat_cols[j]] = valid[cat_cols[i]] ^ valid[cat_cols[j]]

        # 3. x_num_0 * x_cat_9
        train_result['x_num_0_*_x_cat_9'] = train['x_num_0'] * train['x_cat_9']
        valid_result['x_num_0_*_x_cat_9'] = valid['x_num_0'] * valid['x_cat_9']

        # 4. x_num_47_last * x_cat_0
        train_result['x_num_47_last_*_x_cat_0'] = train['x_num_47_last'] * train['x_cat_0']
        valid_result['x_num_47_last_*_x_cat_0'] = valid['x_num_47_last'] * valid['x_cat_0']

        # 5. m表上的 num / sum
        long_cols = ['x_num_13_last', 'x_num_20_last', 'x_num_28_last', 'x_num_30_last',
                     'x_num_43_last', 'x_num_46_last', 'x_num_47_last']

        for col in long_cols:
            train_result[col + '_ratio'] = train[col] / train[long_cols].sum(axis=1)
            valid_result[col + '_ratio'] = valid[col] / valid[long_cols].sum(axis=1)

        # 6. m表上的log(num)
        for col in long_cols:
            train_result[col + '_log'] = np.log(train[col] + 1e-99)
            valid_result[col + '_log'] = np.log(valid[col] + 1e-99)

        return train_result, valid_result

    def create_feature(self, train_b, valid_b, train_m, valid_m, train_y):
        # 在行为表上的特征衍生 mean max median last * - 等
        train_info1, valid_info1 = self.pipeline(train_m, valid_m, more_cols)
        train = pd.merge(train_b, train_info1, on='id')
        valid = pd.merge(valid_b, valid_info1, on='id')

        # 基本信息表和行为表特征的衍生
        train_info2, valid_info2 = self.pipeline2(train, valid)
        train = pd.merge(train, train_info2, on='id')
        valid = pd.merge(valid, valid_info2, on='id')

        train_y = train_y['target']
        train.drop(columns='id', inplace=True)
        valid.drop(columns='id', inplace=True)

        return train, valid, train_y

# 训练模型
def train_model(train, valid, train_y):
    train_data = lgb.Dataset(train, label=train_y)

    lgb_params = {'boosting': 'gbdt',
                  'objective': 'binary',
                  'max_depth': 7,
                  'learning_rate': 0.01,
                  'num_leaves': 50,
                  'metric': ['binary_logloss', 'auc'],
                  'verbose': -1
                  }

    # 模型融合
    my_model1 = lgb.train(lgb_params, train_data, 500)
    my_model2 = lgb.train(lgb_params, train_data, 700)
    my_model3 = lgb.train(lgb_params, train_data, 800)

    score1 = my_model1.predict(valid)
    score2 = my_model2.predict(valid)
    score3 = my_model3.predict(valid)

    prediction = 0.09 * score1 + 0.15 * score2 + 0.76 * score3

    prediction = np.where(prediction < threshold, 0, 1)
    result = pd.DataFrame({'id': valid.index, 'target': prediction})

    
    return result


if __name__ == '__main__':
    path = '../data/'

    train_b, valid_b, train_m, valid_m, train_y = read_file(path)

    train_b, valid_b, train_m, valid_m = original_feature_select(train_b, valid_b, train_m, valid_m)

    createFeature = CreateFeature()

    train, valid, train_y = createFeature.create_feature(train_b, valid_b, train_m, valid_m, train_y)

    result = train_model(train, valid, train_y)
    
    result.to_csv('../data/result.csv', index=0)
