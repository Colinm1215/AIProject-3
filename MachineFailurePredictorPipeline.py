#Team 2 – Jarrett Arredondo, Krystal Grant, Colin Mettler, Chloé Plasse
#April 3, 2023
#CS534-S23-S01 Group Project Assignment #3 - B. Project Development

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)

def main():
    #Adjust pandas options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('expand_frame_repr', False)

    #Read in data
    data = pd.read_csv('ai4i2020.csv')

    #RandomUnderSampler
    failure_count_0, failure_count_1 = data['Machine failure'].value_counts()
    failure_0 = data[data['Machine failure'] == 0]
    failure_1 = data[data['Machine failure'] == 1]
    failure_0_under = failure_0.sample(failure_count_1, random_state=0)
    test_under = pd.concat([failure_0_under,failure_1], axis=0)
    print(test_under)

    #Preprocess data
    #Drop columns not necessary for ML models
    #Convert Type to be used in ML models
    preprocessedData = test_under.drop(['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'], axis=1)
    le = LabelEncoder()
    label = le.fit_transform(preprocessedData['Type'])
    preprocessedData.drop(["Type"], axis=1, inplace=True)
    preprocessedData["Type"] = label

    #Split data into training (70%) and test (30%) set
    X = preprocessedData.drop(['Machine failure'], axis=1)
    y = preprocessedData['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Training---------------------------------------------------------------------------------------------------------------
    #Artificial Neural Network
    mlp_parameters = {
        'hidden_layer_sizes': [(10,), (50,), (100,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }
    gs_mlp = GridSearchCV(MLPClassifier(random_state=1), mlp_parameters, scoring='f1', cv=5)
    gs_mlp.fit(X_train, y_train)
    model_gs_mlp = gs_mlp.best_estimator_
    model_gs_mlp.fit(X_train, y_train)
    y_pred_mlp_train = model_gs_mlp.predict(X_train)
    training_mlp_f1_score = f1_score(y_train, y_pred_mlp_train)

    #Support Vector Machine
    svc_parameters = {
        'kernel': ['linear','poly','rbf','sigmoid'],
        'C': [10,1.0,0.1]
    }
    gs_svc = GridSearchCV(SVC(random_state=2), svc_parameters, scoring='f1', cv=5)
    gs_svc.fit(X_train, y_train)
    model_gs_svc = gs_svc.best_estimator_
    model_gs_svc.fit(X_train, y_train)
    y_pred_svc_train = model_gs_svc.predict(X_train)
    training_svc_f1_score = f1_score(y_train, y_pred_svc_train)

    #Bagging Classifier
    bc_parameters = {
        'n_estimators': [1, 10, 100],
        'max_samples': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 1.0]
    }
    gs_bc = GridSearchCV(BaggingClassifier(random_state=3), bc_parameters, scoring='f1', cv=5)
    gs_bc.fit(X_train, y_train)
    model_gs_bc = gs_bc.best_estimator_
    model_gs_bc.fit(X_train, y_train)
    y_pred_bc_train = model_gs_bc.predict(X_train)
    training_bc_f1_score = f1_score(y_train, y_pred_bc_train)

    #AdaBoost
    ab_parameters = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    gs_ab = GridSearchCV(AdaBoostClassifier(random_state=4), ab_parameters, scoring='f1', cv=5)
    gs_ab.fit(X_train, y_train)
    model_gs_ab = gs_ab.best_estimator_
    model_gs_ab.fit(X_train, y_train)
    y_pred_ab_train = model_gs_ab.predict(X_train)
    training_ab_f1_score = f1_score(y_train, y_pred_ab_train)

    #Random Forest
    rfc_parameters = {
        'n_estimators': [100, 200, 300],
        'max_samples': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'max_depth': [2, 4, 6, 8, 10],
        'criterion': ['gini', 'entropy'],
    }
    gs_rfc = GridSearchCV(RandomForestClassifier(random_state=5), rfc_parameters, scoring='f1', cv=5)
    gs_rfc.fit(X_train, y_train)
    model_gs_rfc = gs_rfc.best_estimator_
    model_gs_rfc.fit(X_train, y_train)
    y_pred_rfc_train = model_gs_rfc.predict(X_train)
    training_rfc_f1_score = f1_score(y_train, y_pred_rfc_train)

    #Data for DataFrame
    ml_model = ["Artificial Neural Networks", "Support Vector Machine", "BaggingClassifier", "AdaBoost", "Random Forest"]
    params = [gs_mlp.best_params_, gs_svc.best_params_, gs_bc.best_params_, gs_ab.best_params_, gs_rfc.best_params_]
    training_scores = [training_mlp_f1_score, training_svc_f1_score, training_bc_f1_score, training_ab_f1_score, training_rfc_f1_score]

    #Create DataFrame to print table
    d = {'ML Trained Model':ml_model, 'Its Best Set of Parameter Values': params, 'Its F1-score on the 5-fold Cross Validation on Training Data (70%)':training_scores}
    TrainingTableData = pd.DataFrame(data = d)
    print(TrainingTableData)
    print()

    max_train = TrainingTableData[TrainingTableData['Its F1-score on the 5-fold Cross Validation on Training Data (70%)'] == TrainingTableData['Its F1-score on the 5-fold Cross Validation on Training Data (70%)'].max()]['ML Trained Model']
    # print("The ML model that produced the best F1-score on the training data is: " + str(max.iloc[0][0]))
    print(str(max_train))
#-----------------------------------------------------------------------------------------------------------------------

#Testing----------------------------------------------------------------------------------------------------------------
    # Artificial Neural Network
    y_pred_mlp_test = model_gs_mlp.predict(X_test)
    testing_mlp_f1_score = f1_score(y_test, y_pred_mlp_test)

    # Support Vector Machine
    y_pred_svc_test = model_gs_svc.predict(X_test)
    testing_svc_f1_score = f1_score(y_test, y_pred_svc_test)

    # Bagging Classifier
    y_pred_bc_test = model_gs_bc.predict(X_test)
    testing_bc_f1_score = f1_score(y_test, y_pred_bc_test)

    # AdaBoost
    y_pred_ab_test = model_gs_ab.predict(X_test)
    testing_ab_f1_score = f1_score(y_test, y_pred_ab_test)

    # Random Forest
    y_pred_rfc_test = model_gs_rfc.predict(X_test)
    testing_rfc_f1_score = f1_score(y_test, y_pred_rfc_test)

    testing_scores = [testing_mlp_f1_score, testing_svc_f1_score, testing_bc_f1_score, testing_ab_f1_score, testing_rfc_f1_score]

    TestingTableData = TrainingTableData
    TestingTableData.drop(["Its F1-score on the 5-fold Cross Validation on Training Data (70%)"], axis=1, inplace=True)
    TestingTableData.insert(2,"Its F1-score on Testing Data (30%)",testing_scores,True)
    print(TestingTableData)
    print()

    max_test = TestingTableData[TestingTableData["Its F1-score on Testing Data (30%)"] == TestingTableData["Its F1-score on Testing Data (30%)"].max()]['ML Trained Model']
    # print("The ML model that produced the best F1-score on the training data is: " + str(max.iloc[0][0]))
    print(str(max_test))
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()