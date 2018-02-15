import tensorflow as tf

import numpy as np
import pseudopol.cpseudopol as c_pp


def estimate_with_dnn(trainX, trainY, queryX): 
    trainY=trainY.astype(int)
    trainX=trainX.astype(int)
    queryX=queryX.astype(int)
    MAX=np.max(trainY)
    MIN=np.min(trainY)
    N=MAX-MIN+1
    OFFSET=MIN
    feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(trainX.astype(np.float32))
    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=int(N), feature_columns=feature_cols)
    dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
    dnn_clf.fit(trainX, trainY-OFFSET, batch_size=50, steps=40000)
    return dnn_clf.predict(queryX)['classes']+OFFSET




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


CAPACITY=100

functions={"DNN": estimate_with_dnn}

def evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val):
    results={x:[] for x in functions.keys()}
    sorted_results={x:[] for x in functions.keys()}
    for i in range(2):
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
N, LEARN_CNT, TEST_CNT=10, 10**3, 10**3

#scenario 1: high full rate:
print("High full rate:")
low_val,high_val=5, 24
evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val)

print("Medium full rate:")
low_val,high_val=5, 18
evaluate_scenario(N,LEARN_CNT, TEST_CNT, low_val, high_val)
