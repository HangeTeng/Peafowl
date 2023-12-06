# Primer

## Tested Environment

This code and following instruction is construct on Ubuntu 20.04, with g++ 9.4 and CMake 3.24.

## Install Project and Dependencies
Download the project code and the dependent libraries libOTe and cryptoTools.
```shell
git clone https://github.com/HangeTeng/libSSS
```
Install the libSSS according to the README in the libSSS repository.

Download some other dependencies
```shell
apt install mpich\
cmake\
pybind11-dev\
libgmp-dev\

pip install mpi4py \
torch\
torchvision\
h5py\
pybind11
```

## Compile libSSS and other C lib

Compile libSSS.

Compile the lwr computer libraries.

```shell
# in /src/utils/crypto
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` lwr.cpp -o lwr`python3-config --extension-suffix` && 
cp $(pwd)/lwr.cpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages
```

## Test Scheme

```shell
python dataset.py
```


