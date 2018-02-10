# knapsackai
silly approaches for solving knapsack problem. No significance, only fast checks of ideas.

## Setup/installation:

Run `setup.sh` to install a virtual environment with all needed python modules. Activate it via

    source p3/bin/activate


## random_forest.py

`src/random_forest.py` uses random forests to estimate the maximal possible sum of number `o1...on` with isn't larger than a maximal capacity.

We valuate for n=10, capacity=100 and thee different scenarios where the rate of achieving full capacity is high, medium or low

Average errror of estimation for full_ratio=0.95:

                                  unsorted           sorted
    RandomForestRegressor           0.13              0.08
    RandomForestClassifier          0.10              0.07
    greedy                          4.11              1.75 

The forest learns pretty good to answer almost always 100, and is right in 95 out 100. As expected, the performance is better if objects are sorted - the argument-space is just smaller.

The greedy strategy "biggest first" outperforms random and "biggest last":  1.75:4.11:9.05.

Average errror of estimation for full_ratio=0.55:

                                  unsorted           sorted
    RandomForestRegressor           0.95              0.40
    RandomForestClassifier          1.05              0.43
    greedy                          3.20              1.12 

This is a more interesting scenario, because more difficult for RandomForest. Still better than greedy approach, which gets free ride in 45% of the cases as it always get the right results.

Average errror of estimation for full_ratio=0.01:

                                  unsorted           sorted
    RandomForestRegressor           1.42              0.04
    RandomForestClassifier          2.35              0.03
    greedy                          0.05              0.00 


#### Conclusion:

RandomForestRegressor/Classifier are able to beat the greedy algorithm. However, if different distributions are used for training/testing, so the performance drops considerably.

Sorting values gives a performance-boost.


