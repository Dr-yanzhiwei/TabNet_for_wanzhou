import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.model_selection import GridSearchCV

import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             recall_score, roc_auc_score, roc_curve,
                             f1_score, precision_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import column_or_1d
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from torch.autograd import Variable

warnings.filterwarnings("ignore")


def data_cleaner(data):
    # 完全相同的数据，去掉相同的
    data = data.drop_duplicates()
    # 缺失值去处理，使用均值。还可以用随机森林填补缺失值
    data = data.fillna(data.interpolate())
    # data = data.dropna()              # 删除缺失值
    # data = df_mean.fit_transform(data)
    # 离群点处理：可能是真实数据产生，也可能是噪声带来的
    # 噪点处理：测量变量中的随机误差或者方差
    # print('cleaner: \n', data)
    return data


def data_integration(data):
    # print('pearson: \n', data.corr(method="pearson"))
    # print('pearson: \n', data)
    # 直接在date_info里进行
    pass
    return data


def data_protocol(data):
    pca = PCA(n_components='mle')
    data = pca.fit_transform(data)
    # print('pca: \n', data)
    # print('贡献率:', pca.explained_variance_ratio_)
    return data


def data_conversion(data):
    std = preprocessing.StandardScaler()
    data = std.fit_transform(data)
    # print('std: \n', data)
    return data


def read_csv_wan(filename):
    df = pd.read_csv(f'{filename}.csv')
    df = data_cleaner(df)
    df = df.drop(columns=['A7', 'A8', 'A9', 'A10', 'A12', 'A13', 'A14', 'B1', 'B2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
    target = 'value'
    IDCol = 'ID'
    GeoID = df[IDCol]
    x_columns = [x for x in df.columns if x not in ['value', target, IDCol, 'GRID_CODE', 'class']]
    # print(x_columns)
    X = df[x_columns]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    return X_train, X_test, y_train, y_test, GeoID, X, y


def over_sampling(X, y):
    oversampler = SMOTE(random_state=0)
    X, y = oversampler.fit_resample(X, y)
    return X, y


def pre_por(X):
    # X = pd.DataFrame(X)
    X = data_protocol(X)
    X = data_conversion(X)
    return X


def inputtocuda(input, output, device):
    """
    将数据集的输入和标签转为cuda格式，使用GPU训练使用这种格式的数据集
    先转tensor，再转cuda
    :param inputtensor: 数据集输入
    :param labeltensor: 数据集标签
    :return: 输入tensor，标签tensor
    """

    # inputtensor = input.data.cpu().detach().numpy()
    inputtensor = np.array(input)
    inputtensor = torch.tensor(inputtensor, dtype=torch.float32)

    # labeltensor = output.data.cpu().detach().numpy()
    labeltensor = np.array(output)
    labeltensor = torch.tensor(labeltensor, dtype=torch.int64)

    inputcuda = Variable(inputtensor).cuda(device)
    outputcuda = Variable(labeltensor).cuda(device)

    return inputcuda, outputcuda


def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def print_result(y_test, y_pred):
    print("======================================================================================")
    print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - Accuracy {:.3f}'
          ' - F1_score {:.3f} - precision {:.3f} '
          .format(balanced_accuracy_score(y_test, y_pred),
                  geometric_mean_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  accuracy_score(y_test, y_pred),
                  f1_score(y_test, y_pred, average='binary'),
                  precision_score(y_test, y_pred, average='binary')))
    print("======================================================================================")


def save_results(GeoID, y_pred, y_predprob, result_file):
    results = np.vstack((GeoID, y_pred, y_predprob))
    results = np.transpose(results)
    header_string = 'GeoID, y_pred, y_predprob'
    np.savetxt(result_file, results, header=header_string, fmt='%d,%d,%0.5f', delimiter=',')
    print('Saving file Done!')


def plot_roc(X_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_lr, tpr_lr, label='LR(AUC=%0.3f)' % auc_lr, lw=2)
    # plt.plot(fpr_lgb, tpr_lgb, label='LGBM(AUC=%0.3f)' % auc_lgb, lw=2)
    # plt.plot(fpr_rf, tpr_rf, label='RF(AUC=%0.3f)' % auc_rf, lw=2)
    # plt.plot(fpr_tn, tpr_tn, label='TN(AUC=%0.3f)' % auc_tn, lw=2)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()


def train(clf, X_train, X_test, y_train, y_test, val_x, val_y):
    if clf == tn:
        print("Training...")
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (val_x, val_y)],
            eval_name=['train', 'valid'],
            eval_metric=['logloss', 'auc'],
            max_epochs=100,  # 最大迭代次数
            patience=20,  # 在验证集上早停次数，
            batch_size=128,  # BN作用在的输入特征batch
            virtual_batch_size=16,  # 除了作用于模型输入特征的第一层BN外，都是用的是ghost BN。
            num_workers=0,
            drop_last=False
        )
        print("predicting...")
        # 特征重要性
        # print(clf.feature_importances_)
        # plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
        # plt.show()
        # exit()
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        print(clf)
        print_result(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
    else:
        print("Training...")
        clf.fit(X_train, y_train)
        # 特征重要性
        # lr
        # n = clf.coef_
        # print(n)
        # rf and lgbm
        # print(clf.feature_importances_)
        # plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
        # plt.show()
        # exit()
        print("predicting...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        # ROC,AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        print(clf)
        print_result(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
    print("======================================Finish!========================================")
    return fpr, tpr, auc, clf


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, GeoID, X, y = read_csv_wan('.\\Raw data\\wanzhou_all')
    X_train, y_train = over_sampling(X_train, y_train)
    # X_train = pre_por(X_train)
    # X_test = pre_por(X_test)
    X = X.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_train, val_x, y_train, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    lr = LogisticRegressionCV(cv=5, random_state=0)
    rf = RandomForestClassifier(random_state=0)
    lgb = LGBMClassifier(random_state=0, n_jobs=-1)
    tn = TabNetClassifier(
        input_dim=16,
        n_steps=4,  # 决策步的个数。可理解为决策树中分裂结点的次数
        n_d=16,  # 可以理解为用来决定输出的隐藏层神经元个数。n_d越大，拟合能力越强，也容易过拟合
        n_a=16,  # 可以理解为用来决定下一决策步特征选择的隐藏层神经元个数
        gamma=1.2,  # 决定历史所用特征在当前决策步的特征选择阶段的权重，gamma=1时，表示每个特征在所有决策步中至多仅出现1次
        momentum=0.02,
        lambda_sparse=1e-3,  # 稀疏正则项权重，用来对特征选择阶段的特征稀疏性添加约束,越大则特征选择越稀疏
        optimizer_fn=torch.optim.Adam,  # 优化器
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type="entmax",
        seed=0
    )

    # 获取tabnet最优参数
    # params = {'gamma': 1.2, 'momentum': 0.02, 'n_steps': 4}

    # params_grid = {
    #     'n_steps': [3, 4, 5, 6, 7, 8, 9, 10],
    #     'gamma': [1.2, 1.3],
    #     'momentum': [0.01, 0.02]
    # }

    # grid = GridSearchCV(tn, params_grid, cv=5, scoring='neg_mean_squared_error')
    # print("开始训练")
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)

    # 五个TN模型的对比
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    #
    # for i in range(5):
    #     if i == 0:  # 采样预处理
    #         print("1：采样预处理")
    #         X_train, X_test, y_train, y_test = read_csv_wan('.\\Raw data\\wanzhou_all')
    #         X_train, y_train = over_sampling(X_train, y_train)
    #         X_train = pre_por(X_train)
    #         X_test = pre_por(X_test)
    #
    #     if i == 1:  # 预处理采样
    #         print("2：预处理采样")
    #         X_train, X_test, y_train, y_test = read_csv_wan('.\\Raw data\\wanzhou_all')
    #         X_train = pre_por(X_train)
    #         X_test = pre_por(X_test)
    #         X_train, y_train = over_sampling(X_train, y_train)
    #     if i == 2:  # 仅预处理
    #         print("3：仅预处理")
    #         X_train, X_test, y_train, y_test = read_csv_wan('.\\Raw data\\wanzhou_all')
    #         X_train = pre_por(X_train)
    #         X_test = pre_por(X_test)
    #     if i == 3:  # 仅采样
    #         print("4：仅采样")
    #         X_train, X_test, y_train, y_test = read_csv_wan('.\\Raw data\\wanzhou_all')
    #         X_train, y_train = over_sampling(X_train, y_train)
    #         X_train = X_train.to_numpy()
    #         X_test = X_test.to_numpy()
    #     if i == 4:  # 不采取操作
    #         print("5：不采取操作")
    #         X_train, X_test, y_train, y_test = read_csv_wan('.\\Raw data\\wanzhou_all')
    #         X_train = X_train.to_numpy()
    #         X_test = X_test.to_numpy()
    #     X_train, val_x, y_train, val_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    #
    #     fpr_tn, tpr_tn, auc_tn = train(tn, X_train, X_test, y_train, y_test, val_x, val_y)
    #
    #     plt.plot(fpr_tn, tpr_tn, label='TN_%d(AUC=%0.3f)' % (i+1, auc_tn), lw=2)

    # 保存结果 and tarin

    fpr_lr, tpr_lr, auc_lr, clf_lr = train(lr, X_train, X_test, y_train, y_test, None, None)
    # y_pred_lr = clf_lr.predict(X)
    # y_pred_proba_lr = clf_lr.predict_proba(X)[:, 1]
    # result_file_lr = './Model and Result/lr.txt'
    # save_results(GeoID, y_pred_lr, y_pred_proba_lr, result_file_lr)

    # fpr_rf, tpr_rf, auc_rf, clf_rf = train(rf, X_train, X_test, y_train, y_test, None, None)
    # y_pred_rf = clf_rf.predict(X)
    # y_pred_proba_rf = clf_rf.predict_proba(X)[:, 1]
    # result_file_rf = './Model and Result/rf.txt'
    # save_results(GeoID, y_pred_rf, y_pred_proba_rf, result_file_rf)

    # fpr_lgb, tpr_lgb, auc_lgb, clf_lgb = train(lgb, X_train, X_test, y_train, y_test, None, None)
    # y_pred_lgb = clf_lgb.predict(X)
    # y_pred_proba_lgb = clf_lgb.predict_proba(X)[:, 1]
    # result_file_lgb = './Model and Result/lgb.txt'
    # save_results(GeoID, y_pred_lgb, y_pred_proba_lgb, result_file_lgb)
    # fpr_tn, tpr_tn, auc_tn, clf_tn = train(tn, X_train, X_test, y_train, y_test, val_x, val_y)
    # tn.fit(
    #     X_train=X_train, y_train=y_train,
    #     eval_set=[(X_train, y_train), (val_x, val_y)],
    #     eval_name=['train', 'valid'],
    #     eval_metric=['logloss', 'auc'],
    #     max_epochs=100,  # 最大迭代次数
    #     patience=20,  # 在验证集上早停次数，
    #     batch_size=128,  # BN作用在的输入特征batch
    #     virtual_batch_size=16,  # 除了作用于模型输入特征的第一层BN外，都是用的是ghost BN。
    #     num_workers=0,
    #     drop_last=False
    # )
    # print("predicting...")
    # y_pred_tn = tn.predict(X)
    # y_pred_proba_tn = tn.predict_proba(X)[:, 1]
    # result_file_tn = './Model and Result/tn.txt'
    # save_results(GeoID, y_pred_tn, y_pred_proba_tn, result_file_tn)
    # print("Donee!!")
    # exit()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label='LR(AUC=%0.3f)' % auc_lr, lw=2)
    # plt.plot(fpr_rf, tpr_rf, label='RF(AUC=%0.3f)' % auc_rf, lw=2)
    # plt.plot(fpr_lgb, tpr_lgb, label='LGBM(AUC=%0.3f)' % auc_lgb, lw=2)
    # plt.plot(fpr_tn, tpr_tn, label='TN(AUC=%0.3f)' % auc_tn, lw=2)

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()

    print("Done!")
