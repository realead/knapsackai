
import numpy as np
import pseudopol.cpseudopol as c_pp

from random_forest import estimate_with_random_forest_regressor, estimate_with_random_forest_classifier
from greedy import estimate_with_greedy

CAPACITY=100

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
functions={"RandomForestRegressor":  estimate_with_random_forest_regressor,
           "RandomForestClassifier": estimate_with_random_forest_classifier,
           "greedy":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,False),
           "greedy_reversed":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,True)}
    

def evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val):
    results={x:[] for x in functions.keys()}
    sorted_results={x:[] for x in functions.keys()}
    for i in range(1):
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





