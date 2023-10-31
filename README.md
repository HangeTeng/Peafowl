torch
torchvision
h5py

pip install pybind11

apt install mpich
pip install mpi4py

apt install cmake
apt install pybind11-dev
apt install libgmp-dev

/src/utils/crypto
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` lwr.cpp -o lwr`python3-config --extension-suffix` && 

cp $(pwd)/lwr.cpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages