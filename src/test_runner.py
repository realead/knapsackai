
from collections import namedtuple

import numpy as np
import pseudopol.cpseudopol as c_pp

from random_forest import estimate_with_random_forest_regressor, estimate_with_random_forest_classifier
from greedy import estimate_with_greedy

def evaluate_result(expectedY, receivedY):
   error=np.abs(expectedY-receivedY)
   return np.sum(error)/error.size
   #return (np.sum(error)/error.size, np.max(error))
   

def get_XY(sc, case_cnt):
    trainX=np.random.randint(sc.low_value, sc.high_value, (case_cnt, sc.object_cnt), dtype=np.uint32)
    trainY=np.zeros((case_cnt,), dtype=np.float)#Classifer seems to need it
    for i in range(case_cnt):
        trainY[i]=c_pp.find_max_subsum(sc.capacity, trainX[i,:])
    return trainX, trainY
  

def evaluate_scenario(scenario, funs):
    results={x:[] for x in funs.keys()}
    sorted_results={x:[] for x in funs.keys()}
    for i in range(1):
        trainX, trainY=get_XY(scenario, scenario.learn_samples_cnt)
        testX,   testY=get_XY(scenario, scenario.test_samples_cnt)
        print("full ratio: %f"%(np.count_nonzero(trainY==CAPACITY)/trainY.size))
        for name,fun in funs.items():
            results[name].append(evaluate_result(testY, fun(trainX, trainY, testX)))
        trainX.sort(axis=1)
        testX.sort(axis=1)
        for name,fun in funs.items():
            sorted_results[name].append(evaluate_result(testY, fun(trainX, trainY, testX)))   
    print("unsorted:",  ",".join(["%s = %f"%(n, sum(l)/len(l)) for n,l in        results.items()]))
    print("sorted:"  ,  ",".join("%s = %f"%(n, sum(l)/len(l)) for n,l in sorted_results.items()))


#### 

#call them with fun(trainx, trainy, testx) to get testy
strategies={"RandomForestRegressor":  estimate_with_random_forest_regressor,
           "RandomForestClassifier": estimate_with_random_forest_classifier,
           "greedy":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,False),
           "greedy_reversed":lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,True)}

Scenario = namedtuple('Scenario', ['capacity', 'object_cnt', 'learn_samples_cnt', 'test_samples_cnt', 'low_value', 'high_value'])


CAPACITY=100
OBJECT_CNT, LEARN_CNT, TEST_CNT=10, 10**3, 10**3

scenarios={"high"   : Scenario(CAPACITY, OBJECT_CNT, LEARN_CNT,TEST_CNT, low_value=5, high_value=24),
           "medium" : Scenario(CAPACITY, OBJECT_CNT, LEARN_CNT,TEST_CNT, low_value=5, high_value=18),
           "low"    : Scenario(CAPACITY, OBJECT_CNT, LEARN_CNT,TEST_CNT, low_value=5, high_value=13)
          }


#evaluation loop:
for name,sc in scenarios.items():
   print("Name:", name)
   evaluate_scenario(sc, strategies)


