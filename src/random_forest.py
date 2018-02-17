from sklearn.ensemble.forest import RandomForestRegressor as RFR
from sklearn.ensemble.forest import RandomForestClassifier as RFC



def estimate_with_rf(trainX, trainY, queryX, rfclass=RFR):  
    max_depth = 35
    regr_rf = rfclass(max_depth=max_depth,n_estimators=25)
    regr_rf.fit(trainX, trainY)
    return regr_rf.predict(queryX)


def estimate_with_random_forest_regressor(trainX, trainY, queryX):  
    return estimate_with_rf(trainX, trainY, queryX, rfclass=RFR) 

def estimate_with_random_forest_classifier(trainX, trainY, queryX):  
    return estimate_with_rf(trainX, trainY, queryX, rfclass=RFC)  


