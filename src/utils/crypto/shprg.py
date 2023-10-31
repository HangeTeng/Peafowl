# Creative Commons Zero v1.0 Universal
# SPDX-License-Identifier: CC0-1.0
# Created by Douglas Stebila

import math
import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np
from lwr import lwr_128_64 as lwr

from line_profiler import LineProfiler

# This is for timing


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result

    return func_wrapper


class SHPRG(object):
    def __init__(self,
                 input=2,
                 output=4,
                 EQ=64,
                 EP=32,
                 seedA=bytes(0x355678),
                 load_A=''):
        """To construct a shprg with inputsize = input*EQ bit and outputsize = output * EP bit"""
        self.gen = self.__genAES128
        self.shake = self.__shake128
        # setParams
        self.n = input
        self.m = output
        self.q = 2**EQ
        self.p = 2**EP
        self.EQ = EQ
        self.EP = EP
        self.len_seedAES = 128
        self.len_q_bytes = int(EQ / 8)
        self.len_p_bytes = int(EP / 8)
        self.len_seedAES_bytes = int(self.len_seedAES / 8)

        if load_A == '':
            self.A = self.gen(self.shake(seedA, self.len_seedAES_bytes))
            self.A = np.array(self.A)
            np.save(
                './data/prg/A_n{}_m{}_q{}_p{}.npy'.format(
                    input, output, EQ, EP), self.A)
        else:
            self.A = np.load(load_A, allow_pickle=True)

    @staticmethod
    def __shake128(msg, digest_len):
        """Returns a bytes object containing the SHAKE-128 hash of msg with 
        digest_len bytes of output"""
        shake_ctx = hashes.Hash(hashes.SHAKE128(digest_len),
                                backend=default_backend())
        shake_ctx.update(msg)
        return shake_ctx.finalize()

    @staticmethod
    def __aes128_16bytesonly(key, msg):
        """Returns a bytes object containing the AES-128 encryption of the 16-byte 
        message msg using the given key"""
        cipher_ctx = Cipher(algorithms.AES(key),
                            modes.ECB(),
                            backend=default_backend())
        encryptor_ctx = cipher_ctx.encryptor()
        return encryptor_ctx.update(msg) + encryptor_ctx.finalize()

    # @timer
    def __genAES128(self, seedA):
        """Generate matrix A using AES-128 (SHPRG specification, Algorithm 7)"""
        A = [[0 for _ in range(self.m)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                b = bytearray(16)
                struct.pack_into('<H', b, 0, i % 65536)
                struct.pack_into('<H', b, 2, j % 65536)
                for k in range(math.ceil(self.len_q_bytes / 16)):
                    c = SHPRG.__aes128_16bytesonly(seedA, b)
                    A[i][j] += int.from_bytes(c,
                                              byteorder='big') * (2**(k * 128))
                    A[i][j] %= self.q
        return A

    @staticmethod
    def genMatrixAES128(seed=bytes(0x355678), n=2, m=4, EQ=64):
        """Generate matrix A using AES-128 (SHPRG specification, Algorithm 7)"""
        q = 2**EQ
        len_q_bytes = int(EQ / 8)
        A = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                b = bytearray(16)
                struct.pack_into('<H', b, 0, i % 65536)
                struct.pack_into('<H', b, 2, j % 65536)
                for k in range(math.ceil(len_q_bytes / 16)):
                    c = SHPRG.__aes128_16bytesonly(seed, b)
                    A[i][j] += int.from_bytes(c,
                                              byteorder='big') * (2**(k * 128))
                    A[i][j] %= q
        return A

    # from numba import jit
    # @jit
    def _operation(self, x, p, q):
        result = (x % p * p) // q % p
        return result

    # @profile
    def genRandom(self, s_np):
        A = self.A
        p = self.p
        q = self.q

        if q == 2**128 and p == 2**64:
            return self.genRandom_128_64(s_np)

        output = np.empty((s_np.shape[0], A.shape[1]), dtype=object)
        np.dot(s_np, A, out=output)
        # np.mod(output, q, out=output)
        np.multiply(output, p, out=output)
        np.floor_divide(output, q, out=output)
        np.mod(output, p, out=output)
        return output

    def genRandom_bitop(self, s_np):
        A = self.A
        output = np.empty((s_np.shape[0], A.shape[1]), dtype=object)
        output[...] = s_np @ A
        output <<= EP
        output >>= EQ
        output &= (1 << EP) - 1
        return output

    def genRandom_128_64(self, s_np):
        A = self.__split_array(self.A)
        # print(A)
        s_np_64 = self.__split_array(s_np)
        # print(s_np_64)
        return np.array(lwr(s_np_64, A),dtype=np.uint64)

    def __split_array(self, arr, split_num=2):
        shape = arr.shape
        arr_split = np.empty(shape + (split_num, ), dtype=np.uint64)
        for i in range(split_num):
            arr_split[..., i] = (arr >> (i * 64)) & 0xFFFFFFFFFFFFFFFF
        return arr_split
    


if __name__ == "__main__":
    n = 1
    m = 1
    EQ = 128
    EP = 64
    # load_A = './data/prg/A_n{}_m{}_q{}_p{}.npy'.format(n, m, EQ, EP)
    load_A = ""
    prg = SHPRG(input=n, output=m, EQ=EQ, EP=EP, load_A=load_A)
    s1 = np.array([[i + 1 for i in range(n)]] * 1, dtype=np.uint64)
    print(s1.shape)
    print(prg.A.shape)
    s2 = np.array([[78 for _ in range(n)]]*1, dtype=np.uint64)
    s3 = np.array([[i+1+78 for i in range(n)]]*1, dtype=np.uint64)

    s1 = np.array([[265197659641773038295820696019437708809]],dtype=object)
    s2 = np.array([[75084707279165425167553911412330502767]],dtype=object)
    s3 = np.array([[100], [110], [120], [130], [140]],dtype=np.uint64)

    # print(s1)
    # print(s2)
    # print(s3)
    # print(prg.A)

    def mod_range(a, p):
        "convert to  [-p/2, p/2 - 1]"
        r = (a % p) - p // 2
        return r.astype(np.int64)
    p =2**EP

    # a = prg.genRandom(s1)
    # # print(a)
    # a = prg.genRandom2(s1)
    # # print(a)
    # a = prg.genRandom3(s1)
    # print(a)
    a = prg.genRandom(s1)
    b = prg.genRandom(s2)
    c = prg.genRandom(s3)

    print(a)
    print(mod_range(a, p))
    print(b)
    print(mod_range(b, p))
    print(a+b)
    print(mod_range(a+b, p))
    print()
    print(c)
    print(mod_range(c, p))
    for i in range(m):
        x = mod_range(a+b, p) - mod_range(c, p)
        print(x)
        if(x != 2**EP-1 and x>1):
            print("error!")
