#stop on error
set -e
set -x

##set needed environment:
virtualenv -p python3 p3
. p3/bin/activate

##install needed: 
pip install cython
pip install numpy
pip install scipy
pip install -U scikit-learn
#pip install matplotlib
pip install tensorflow
pip install https://github.com/realead/pseudopol/zipball/master

#show current setup
pip freeze
