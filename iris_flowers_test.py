import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle


def data_load():
    """
    The function load the iris flowers dataset, and create pandas dataframe for better viewing and processing.
    Returns: Shuffled Dataset dataframe.
    """
    iris = load_iris()
    iris_data_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    iris_classification_dataset = pd.DataFrame(data=iris['target'], columns=['iris_classification_type'])
    iris_df = iris_classification_dataset.__deepcopy__()

    for i in range(3):
        iris_classification_dataset = iris_classification_dataset.replace(i, iris['target_names'][i])

    # the merged_df_with_name object is a dataframe with all the iris data and classification.
    merged_df_with_name = pd.concat([iris_data_df, iris_classification_dataset], axis=1)
    # the merged_df_with_num object is a dataframe with all the iris data and number classification.
    merged_df_with_num = pd.concat([iris_data_df, iris_df], axis=1)

    shuffled_df = shuffle(merged_df_with_num)
    print(f'Shuffled DataFrame:\n{shuffled_df}')
    return shuffled_df


def split_data_train_test(iris_df, test_ratio=0.20):
    """
    Split the df to the data and iris classification, and for 20% test and 80% training set.
    Args:
        test_ratio: test set ratio.
        iris_df: The iris dataset DF.

    Returns: x,y train and test set.
    """
    # split to data and labels
    x_df = iris_df.drop(['iris_classification_type'], axis=1)
    y_df = iris_df['iris_classification_type']

    # split to 20% test set and 80% training set.
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_ratio)
    return X_train, X_test, y_train, y_test


def svm_algo_train(X_train, y_train):
    """
    Support Vector Machines algorithm creation and training.
    Args:
        X_train: The X train features.
        y_train: The Y train labels.

    Returns: SVM model.
    """
    model = SVC(gamma='auto')
    model.fit(X_train, y_train)
    return model


def svm_predict(svm_model, X_test, y_test):
    """
    SVM algorithm predict function.

    Args:
        X_test: The test set.
        y_test: The test set labels.

    Returns:
    """
    prediction = svm_model.predict(X_test)
    print(f'SVM Test Accuracy: {accuracy_score(y_test, prediction) * 100}%')
    print(f'SVM Classification Report: \n {classification_report(y_test, prediction)}')


def knn_algo_train(X_train, y_train, neighbors=3):
    """
    k-Nearest Neighbors algorithm create and model fitting.
    Args:
        X_train: The training set.
        y_train: Training set labels.
        neighbors: Number of algo neighbors.

    Returns: KNN model
    """
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model


def knn_predict(knn, X_test, y_test):
    """
    KNN algorithm predict function.
    Args:
        knn: The KNN model.
        X_test: The test set data.
        y_test: The test set labels.

    Returns:
    """
    predictions = knn.predict(X_test)
    print(f'KNN Test Accuracy: {accuracy_score(y_test, predictions) * 100}%')
    print(f'KNN Classification Report: \n {classification_report(y_test, predictions)}')


if __name__ == '__main__':
    # Loading the iris dataframe.
    iris_df = data_load()
    # Split the dataframe.
    X_train, X_test, y_train, y_test = split_data_train_test(iris_df)
    # SVM model
    svm_model = svm_algo_train(X_train, y_train)
    svm_predict(svm_model, X_test, y_test)
    # KNN model
    knn = knn_algo_train(X_train, y_train)
    knn_predict(knn, X_test, y_test)
