import tensorflow as tf

import numpy as np

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



