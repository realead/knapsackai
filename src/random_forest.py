import numpy as np
import pseudopol.cpseudopol as c_pp

from sklearn.ensemble.forest import RandomForestRegressor as RFR
from sklearn.ensemble.forest import RandomForestClassifier as RFC


CAPACITY=100


def estimate_with_rf(trainX, trainY, queryX, rfclass=RFR):  
    max_depth = 35
    regr_rf = rfclass(max_depth=max_depth,n_estimators=25)
    regr_rf.fit(trainX, trainY)
    return regr_rf.predict(queryX)

def estimate_with_greedy(trainX, trainY, queryX, cap, reverse=False):
    def greedy(vals, cap, rev=True):
       res=0;
       v=reversed(vals) if rev else vals
       for i in v:
            if res+i<=cap:
                res+=i
       return res;
    n=queryX.shape[0]
    result=np.zeros((n,))
    for i in range(n):
        result[i]=greedy(queryX[i,:],cap, reverse)
    return result

def evaluate_result(expectedY, receivedY):
   error=np.abs(expectedY-receivedY)
   return np.sum(error)/error.size
   #return (np.sum(error)/error.size, np.max(error))
   

def get_XY(object_cnt, case_cnt, low_val, high_val):
    trainX=np.random.randint(low_val, high_val, (case_cnt, object_cnt), dtype=np.uint32)
    trainY=np.zeros((case_cnt,), dtype=np.float)#Classifer seems to need it
    for i in range(case_cnt):
        trainY[i]=c_pp.find_max_subsum(CAPACITY, trainX[i,:])
    return trainX, trainY


#call them with fun(trainx, trainy, testx) to get testy
functions={"RandomForestRegressor": estimate_with_rf,
           "RandomForestClassifier": lambda x,y,q:estimate_with_rf(x,y,q, RFC),
           "greedy":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,False),
           "greedy_reversed":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,True)}
    

def evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val):
    results={x:[] for x in functions.keys()}
    sorted_results={x:[] for x in functions.keys()}
    for i in range(10):
        trainX, trainY=get_XY(N,LEARN_CNT, low_val, high_val)
        testX, testY=get_XY(N, TEST_CNT, low_val, high_val)
        print("full ratio: %f"%(np.count_nonzero(trainY==CAPACITY)/trainY.size))
        for name,fun in functions.items():
            results[name].append(evaluate_result(testY, fun(trainX, trainY, testX)))
        trainX.sort(axis=1)
        testX.sort(axis=1)
        for name,fun in functions.items():
            sorted_results[name].append(evaluate_result(testY, fun(trainX, trainY, testX)))   
    print("unsorted:",  ",".join(["%s = %f"%(n, sum(l)/len(l)) for n,l in        results.items()]))
    print("sorted:"  ,  ",".join("%s = %f"%(n, sum(l)/len(l)) for n,l in sorted_results.items()))


#### 
N, LEARN_CNT, TEST_CNT=10, 10**5, 10**3

#scenario 1: high full rate:
print("High full rate:")
low_val,high_val=5, 24
evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val)

print("Medium full rate:")
low_val,high_val=5, 18
evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val)

print("Low full rate:")
low_val,high_val=5, 13
evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val)


