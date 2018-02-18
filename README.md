# knapsackai
silly approaches for solving knapsack problem. No significance, only fast checks of ideas.

## Problem:

Estimate the maximal possible sum of number `o1...on` which isn't larger than a maximal capacity.

We valuate for n=10, capacity=100 and thee different scenarios where the rate of achieving full capacity is high, medium or low

## Setup/installation:

Run `setup.sh` to install a virtual environment with all needed python modules. Activate it via

    source p3/bin/activate

##Running tests 

Run `src/test_runner.py` for comparing different approaches for different scenarios. For example

    python test_runner.py --dnn --forest_regressor --high_ratio --medium_ratio --low_ratio

will compare deep neuronal networks with random forest regressor for three different scenarios: high, medium and low.


## random forests


For number of samples for learning=10**5:


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


### Reducing the number of samples for learning

As expected more learn samples means better results, here for RandomForestRegressor(sorted) number of learning samples vs average error

     
     ratio:                100           1000        10**4      10**5
     high                  0.15          0.09        0.08        0.07
     medium                0.93          0.71        0.56        0.42
     low                   1.12          0.58        0.27        0.04

### Conclusion:

RandomForestRegressor/Classifier are able to beat the greedy algorithm. However, if different distributions are used for training/testing, so the performance drops considerably.

Sorting values gives a performance-boost.


## DNN

Deep neuronal networks are much slower to train than the random forests, thus we only do it with 10**3 learn-samples.

Here are the average errors, comparing dnn and forest-regressor


     ratio:                dnn_unsorted       dnn_sorted   forest_unsorted    forest_sorted 
     high                      0.15              0.16          0.20              0.11
     medium                    1.91              0.97          1.41              0.72
     low                       5.50              0.50          2.59              0.58

#### Conclusion:

more experiments needed.


## Trouble shooting:

tensorflow seems to be a moving target right now, the experiments were done with the following set-up:

    >>> pip freeze
    absl-py==0.1.10
    bleach==1.5.0
    cycler==0.10.0
    Cython==0.27.3
    html5lib==0.9999999
    Markdown==2.6.11
    matplotlib==2.1.2
    numpy==1.14.0
    protobuf==3.5.1
    pseudopol==0.1.0
    pyparsing==2.2.0
    python-dateutil==2.6.1
    pytz==2018.3
    scikit-learn==0.19.1
    scipy==1.0.0
    six==1.11.0
    tensorflow==1.5.0
    tensorflow-tensorboard==1.5.1
    Werkzeug==0.14.1



