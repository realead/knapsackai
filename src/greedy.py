import numpy as np

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

