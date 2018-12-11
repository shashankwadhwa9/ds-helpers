import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_null_value_counts(df):
    """
    Returns a Series having the counts of (non zero ) null values of each  column

    :param df: dataframe
    :return: A series with index as the column names and values as the null values in those columns
    """
    null_value_stats = df.isnull().sum(axis=0)
    return null_value_stats[null_value_stats != 0]


def plot_categorical_column(df, target_variable, column_name, fig_size=(9, 12), y_lim=None):
    """
    A generic function to plot the distribution of a categorical column, and
    the ratio of target variable in each of the values of that column.

    :param df: Dataframe
    :param target_variable: Target variable in the dataframe
    :param column_name: Column name
    :param fig_size: Figure size
    :param y_lim: Limit on y-axis
    :return: A grid of two plots
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=fig_size)
    tmp_df = df.dropna()
    sns.countplot(x=column_name, data=tmp_df, ax=ax1)
    sns.pointplot(x=column_name, y=target_variable, data=tmp_df, ax=ax2)
    if y_lim is not None:
        ax2.set_ylim(0, y_lim)


def plot_continuous_column(df, target_variable, column_name, fig_size=(9, 12)):
    """
    A generic function to plot the distribution of a continuous column, and
    boxplot of that column for each value of target_variable

    :param df: Dataframe
    :param target_variable: Target variable in the dataframe
    :param column_name: Column name
    :param fig_size: Figure size
    :return: A grid of two plots
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=fig_size)
    tmp_df = df.dropna()
    sns.distplot(tmp_df[column_name], ax=ax1)
    sns.boxplot(x=target_variable, y=column_name, data=tmp_df, ax=ax2)


def evaluate_models(
        df, features, target_variable, test_size=0.2, seed=1, lr=True, lda=True, svm=False, knn=True, nb=True, dt=True,
        rf=True, et=True, gb=True, ada=True, nn=True, scoring_metric='accuracy'
):
    """
    Evaluate different models on the passed dataframe using the given features.
    """

    # Create testing and training data
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target_variable], test_size=test_size, random_state=seed
    )

    results = {}  # to store the results of the models
    models = []
    scoring_dict = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }
    scoring_func = scoring_dict.get(scoring_metric)

    if lr:
        models.append(('lr', LogisticRegression(random_state=seed)))
    if lda:
        models.append(('lda', LinearDiscriminantAnalysis()))
    if svm:
        models.append(('svm', SVC(random_state=seed)))
    if knn:
        models.append(('knn', KNeighborsClassifier(n_neighbors=5)))
    if nb:
        models.append(('nb', GaussianNB()))
    if dt:
        models.append(('dt', DecisionTreeClassifier(random_state=seed)))
    if rf:
        models.append(('rf', RandomForestClassifier(random_state=seed, n_estimators=100)))
    if et:
        models.append(('et', ExtraTreesClassifier(random_state=seed, n_estimators=100)))
    if gb:
        models.append(('gb', GradientBoostingClassifier(random_state=seed, n_estimators=100)))
    if ada:
        models.append(('ada', AdaBoostClassifier(random_state=seed)))
    if nn:
        models.append(('nn', MLPClassifier(random_state=seed)))

    for model_name, model in models:
        print(f'Running {model_name}')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score_value = scoring_func(y_test, y_pred)
        print(f'Score value for {model_name} is {score_value}')
        results[model_name] = (model, score_value)

    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for model_name, (model, score_value) in sorted_results:
        print(model_name, score_value)

    return results


def encode_categorical(df, categorical_columns, drop_first=True):
    """
    Returns a new dataframe which has the categorical features converted to dummy variables.

    :param df: Dataframe
    :param categorical_columns: list of categorical features
    :param drop_first: Whether to drop the first dummy variable
    :return: A new dataframe which has the categorical features converted to dummy variables.
    """
    df_copy = df.copy()
    for categorical_column in categorical_columns:
        dummy_df = pd.get_dummies(df_copy[categorical_column], prefix=categorical_column, drop_first=drop_first)
        df_copy = pd.concat([df_copy, dummy_df], axis=1)

    return df_copy


def get_scaled_df(df, features, target_variable, type='standard'):
    """
    Returns a new dataframe in which the columns are scaled according to the type mentioned.

    :param df: Dataframe
    :param features: features
    :param target_variable: Target variable in the dataframe
    :param type: standard/min_max
    :return: A new dataframe in which the columns are scaled according to the type mentioned.
    """
    mapping_dict = {
        'standard': StandardScaler(),
        'min_max': MinMaxScaler()
    }
    scaler = mapping_dict[type]
    scaled_df = pd.DataFrame(scaler.fit_transform(df[features]), index=df.index)
    scaled_df.columns = features
    scaled_df[target_variable] = df[target_variable]
    return scaled_df


def get_feature_importances(model, features):
    """
    Returns a list of sorted feature importances along with the scores

    :param model: Model which hs been trained
    :param features: Features
    :return: List of sorted feature importances along with the scores
    """
    if type(model) == RandomForestClassifier:
        return sorted(zip(model.feature_importances_, features), reverse=True)
    elif type(model) == LogisticRegression:
        return sorted(zip(abs(model.coef_[0]), features), reverse=True)

    return None
