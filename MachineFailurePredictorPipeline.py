#Team 2 – Jarrett Arredondo, Krystal Grant, Colin Mettler, Chloé Plasse
#April 3, 2023
#CS534-S23-S01 Group Project Assignment #3 - B. Project Development

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def main():
    #Read in data
    data = pd.read_csv('ai4i2020.csv')

    #RandomUnderSampler
    failure_count_0, failure_count_1 = data['Machine failure'].value_counts()
    failure_0 = data[data['Machine failure'] == 0]
    failure_1 = data[data['Machine failure'] == 1]
    failure_0_under = failure_0.sample(failure_count_1, random_state=1)
    test_under = pd.concat([failure_0_under,failure_1], axis=0)

    #Preprocess date
    #Drop columns not necessary for ML models
    #Convert Type to be used in ML models
    preprocessedData = test_under.drop(['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'], axis=1)
    le = LabelEncoder()
    label = le.fit_transform(preprocessedData['Type'])
    preprocessedData.drop(["Type"], axis=1, inplace=True)
    preprocessedData["Type"] = label

    #Split data into training and test set
    X = preprocessedData.drop(['Machine failure'], axis=1)
    y = preprocessedData['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Training---------------------------------------------------------------------------------------------------------------
    #Artificial Neural Network
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_train)
    training_mlp_f1_score = f1_score(y_train, y_pred)

    #Support Vector Machine
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_train)
    training_svc_f1_score = f1_score(y_train, y_pred)

    #Bagging Classifier
    bc = BaggingClassifier()
    bc.fit(X_train, y_train)
    y_pred = bc.predict(X_train)
    training_bc_f1_score = f1_score(y_train, y_pred)

    #AdaBoost
    ab = AdaBoostClassifier()
    ab.fit(X_train, y_train)
    y_pred = ab.predict(X_train)
    training_ab_f1_score = f1_score(y_train, y_pred)

    #Random Forest
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_train)
    training_rfc_f1_score = f1_score(y_train, y_pred)

    mlp_params = "(" + str(mlp.hidden_layer_sizes) + ", " + str(mlp.activation) + ")"
    svc_params = "(" + str(svc.C) + ", " + svc.kernel + ")"
    bc_params = "(" + str(bc.n_estimators) + ", " + str(bc.max_samples) + ", " + str(bc.max_features) + ")"
    ab_params = "(" + str(ab.n_estimators) + ", " + str(ab.learning_rate) + ")"
    rfc_params = "(" + str(rfc.n_estimators) + ", " + str(rfc.criterion) + ", " + str(rfc.max_depth) + ", " + str(rfc.max_samples) + ")"

    ml_model = ["Artificial Neural Networks", "Support Vector Machine", "BaggingClassifier", "AdaBoost", "Random Forest"]
    params = [mlp_params, svc_params, bc_params, ab_params, rfc_params]
    training_scores = [training_mlp_f1_score, training_svc_f1_score, training_bc_f1_score, training_ab_f1_score, training_rfc_f1_score]

    d = {'ML Trained Model':ml_model, 'Its Best Set of Parameter Values': params, 'Its F1-score on the 5-fold Cross Validation on Training Data (70%)':training_scores}
    TrainingTableData = pd.DataFrame(data = d)
    pd.set_option('display.max_columns',None)
    print(TrainingTableData)
    print()
#-----------------------------------------------------------------------------------------------------------------------
    max = TrainingTableData[TrainingTableData['Its F1-score on the 5-fold Cross Validation on Training Data (70%)'] == TrainingTableData['Its F1-score on the 5-fold Cross Validation on Training Data (70%)'].max()]['ML Trained Model']
    print("The ML model that produced the best F1-score on the training data is: " + str(max.iloc[0][0]))


    # Testing
    # Artificial Neural Network
    y_pred = mlp.predict(X_test)
    testing_mlp_f1_score = f1_score(y_test, y_pred)

    # Support Vector Machine
    y_pred = svc.predict(X_test)
    testing_svc_f1_score = f1_score(y_test, y_pred)

    # Bagging Classifier
    y_pred = bc.predict(X_test)
    testing_bc_f1_score = f1_score(y_test, y_pred)

    # AdaBoost
    y_pred = ab.predict(X_test)
    testing_ab_f1_score = f1_score(y_test, y_pred)

    # Random Forest
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    testing_rfc_f1_score = f1_score(y_test, y_pred)

    testing_scores = [testing_mlp_f1_score, testing_svc_f1_score, testing_bc_f1_score, testing_ab_f1_score, testing_rfc_f1_score]

    TestingTableData = TrainingTableData
    TestingTableData.drop(["Its F1-score on the 5-fold Cross Validation on Training Data (70%)"], axis=1, inplace=True)
    TestingTableData.insert(2,"Its F1-score on Testing Data (30%)",testing_scores,True)
    print(TestingTableData)
    print()

    # rfc_parameters = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth': [2, 4, 6, 8, 10],
    #     'criterion': ['gini', 'entropy'],
    #     'min_samples_split': [2, 4, 6]
    # }
    #
    # rand_search = RandomizedSearchCV(rfc, rfc_parameters, cv=5)
    # rand_search.fit(X_train, y_train)
    # print(rand_search.best_params_)
    # print("best accuracy :", rand_search.best_score_)

if __name__ == "__main__":
    main()