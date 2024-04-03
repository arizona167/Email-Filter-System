#A project by Arinze Igwegbe and Cherry Peth



import numpy as np
import matplotlib.pyplot as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.covariance import EllipticEnvelope

def load_and_combine_datasets(spam1, spam2):
    data1 = np.loadtxt(spam1, delimiter=',')
    data2 = np.loadtxt(spam2, delimiter=',')
    combined_data = np.concatenate((data1, data2), axis=0)
    return combined_data

def preprocess_data(data):
    # Separate features and labels
    features = data[:, :-1]
    labels = data[:, -1]

    # Outlier detection and removal
    outlier_detector = EllipticEnvelope(contamination=0.1) 
    outlier_mask = outlier_detector.fit_predict(features)
    filtered_features = features[outlier_mask == 1]
    filtered_labels = labels[outlier_mask == 1]

    # Remove rows with missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(filtered_features)

    return features_imputed, filtered_labels

def predictTest(trainFeatures, trainLabels, testFeatures):
    model = RandomForestClassifier()
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:, 1]
    return testOutputs

def aucCV(features, labels):
    model = RandomForestClassifier()
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

if __name__ == "__main__":
    # Load and combine datasets
    combined_data = load_and_combine_datasets('spam1.csv', 'spam2.csv')

    # Preprocess combined data
    preprocessed_features, preprocessed_labels = preprocess_data(combined_data)

    # Split preprocessed data into train and test sets
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(preprocessed_features, preprocessed_labels, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(trainFeatures, trainLabels)
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Evaluating classifier accuracy using 10-fold cross-validation
    model = RandomForestClassifier(**best_params)
    scores = aucCV(trainFeatures, trainLabels)
    print("10-fold cross-validation mean AUC:", np.mean(scores))

    # Train the model
    model.fit(trainFeatures, trainLabels)

    # Predict on the test set
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)

    # Calculate AUC for the test set
    testAUC = roc_auc_score(testLabels, testOutputs)
    print("Test set AUC:", testAUC)

    # Calculate precision, recall, and F1-score
    testPredictions = (testOutputs > 0.5).astype(int)
    precision = precision_score(testLabels, testPredictions)
    recall = recall_score(testLabels, testPredictions)
    f1 = f1_score(testLabels, testPredictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    pl.subplot(2, 1, 1)
    pl.plot(np.arange(nTestExamples), testLabels[sortIndex], 'b.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Target')
    pl.subplot(2, 1, 2)
    pl.plot(np.arange(nTestExamples), testOutputs[sortIndex], 'r.')
    pl.xlabel('Sorted example number')
    pl.ylabel('Output (predicted target)')
    pl.show()
