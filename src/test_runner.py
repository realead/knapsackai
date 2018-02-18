
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
    for i in range(scenario.run_cnt):
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

CAPACITY=100
OBJECT_CNT=10
ValRange = namedtuple('ValRange', ['low_value', 'high_value'])


#call them with fun(trainx, trainy, testx) to get testy
STRATEGIES={"forest_regressor"  :  estimate_with_random_forest_regressor,
            "forest_classifier" : estimate_with_random_forest_classifier,
            "greedy": lambda x,y,q:estimate_with_greedy(x,y,q,CAPACITY,True)
           }

VAL_RANGES={"high_ratio"   : ValRange(low_value=5, high_value=24),
            "medium_ratio" : ValRange(low_value=5, high_value=18),
            "low_ratio"    : ValRange(low_value=5, high_value=13)
           }


###parse command line:
import argparse

parser = argparse.ArgumentParser(description='run tests for different solution-strategies')
parser.add_argument('--n_learn', type=int, default=10**3,
                   help='number of samples for the learning phase')
parser.add_argument('--n_test', type=int, default=10**3,
                   help='number of samples for the testing phase')
parser.add_argument('--n_runs', type=int, default=2,
                   help='number of runs over which the results are averaged')
#strategies
for name in STRATEGIES.keys():
   parser.add_argument('--'+name, dest='strategies', action='append_const', const=name, help='a possible strategy')

#scenarios
for name in VAL_RANGES.keys():
   parser.add_argument('--'+name, dest='scenarios', action='append_const', const=name, help='a possible scenario')


args = parser.parse_args()



#evaluate arguments:

Scenario = namedtuple('Scenario', ['capacity', 'object_cnt', 'learn_samples_cnt', 'test_samples_cnt', 'run_cnt', 'low_value', 'high_value'])


strategies = { name : STRATEGIES[name] for name in args.strategies}
scenarios  = { name : Scenario(CAPACITY, OBJECT_CNT, 
                               learn_samples_cnt=args.n_learn, test_samples_cnt=args.n_test,
                               run_cnt=args.n_runs, 
                               low_value=VAL_RANGES[name].low_value,
                               high_value=VAL_RANGES[name].high_value)
                      for name in args.scenarios}



#evaluation loop:
for name,sc in scenarios.items():
   print("Name:", name)
   evaluate_scenario(sc, strategies)


