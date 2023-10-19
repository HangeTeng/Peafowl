# Creative Commons Zero v1.0 Universal
# SPDX-License-Identifier: CC0-1.0
# Created by Douglas Stebila

import math
import struct
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import numpy as np

# This is for timing
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper

class SHPRG(object):
    def __init__(self, input=2, output =4, EQ = 64, EP = 32, seedA = bytes(0x355678), load_A=""):
        """To construct a shprg with inputsize = input*EQ bit and outputsize = output * EP bit"""
        self.print_intermediate_values = True
        # self.randombytes = lambda k : bytes((secrets.randbits(8) for i in range(k)))
        self.gen = self.genAES128
        self.shake = self.__shake128
        # setParams()
        self.n = input
        self.m = output
        self.q = 2 ** EQ
        self.p = 2 ** EP
        self.len_seedAES = 128
        self.len_q_bytes = int(EQ / 8)
        self.len_p_bytes = int(EP / 8)
        self.len_seedAES_bytes = int(self.len_seedAES / 8)

        if load_A == "":
            self.A = self.gen(self.shake(seedA, self.len_seedAES_bytes))
            # print(self.A)
            self.A = np.array(self.A)
            # print(self.A)
            print(type(self.A[0][0]))
        else: 
            self.A = np.load(load_A)


    def __print_intermediate_value(self, name, value):
        """Prints an intermediate value for debugging purposes"""
        if not(self.print_intermediate_values): return None
        if isinstance(value, bytes):
            print("{:s} ({:d}) = {:s}".format(name, len(value), value.hex().upper()))
        elif name in ["r"]:
            print("{:s} ({:d}) = ".format(name, len(value)), end='')
            for i in range(len(value)):
                print("{:d},".format(value[i] % self.q), end='')
            print()
        elif name in ["A", "B", "B'", "B''", "B'S", "C", "C'", "E", "E'", "E''", "M", "S", "S'", "S^T", "V", "mu_encoded"]:
            print("{:s} ({:d} x {:d}) = ".format(name, len(value), len(value[0])), end='')
            for i in range(len(value)):
                for j in range(len(value[0])):
                    print("{:d},".format(value[i][j] % self.q), end='')
            print()
        else:
            assert False, "Unknown value type for " + name
    
    @staticmethod
    def __shake128(msg, digest_len):
        """Returns a bytes object containing the SHAKE-128 hash of msg with 
        digest_len bytes of output"""
        shake_ctx = hashes.Hash(hashes.SHAKE128(digest_len), backend = default_backend())
        shake_ctx.update(msg)
        return shake_ctx.finalize()

    @staticmethod
    def __aes128_16bytesonly(key, msg):
        """Returns a bytes object containing the AES-128 encryption of the 16-byte 
        message msg using the given key"""
        cipher_ctx = Cipher(algorithms.AES(key), modes.ECB(), backend = default_backend())
        encryptor_ctx = cipher_ctx.encryptor()
        return encryptor_ctx.update(msg) + encryptor_ctx.finalize()

    @timer
    def __matrix_mul_round(self, X, Y):
        """Compute matrix multiplication X * Y mod q"""
        X_np = np.array(X, dtype=np.float64)
        Y_np = np.array(Y, dtype=np.float64)
        R = np.dot(X_np, Y_np)* self.p / self.q + 0.5
        print(type(self.q))
        print(self.q)
        R = np.floor(R) % self.p
        
        print(type(R[0][0]))
        R = R.tolist()
        print(type(R[0][0]))
        return R
    
    @timer
    def genAES128(self, seedA):
        """Generate matrix A using AES-128 (SHPRG specification, Algorithm 7)"""
        A = [[0 for j in range(self.n)] for i in range(self.m)]
        for i in range(self.m):
            for j in range(self.n):
                b = bytearray(16)
                struct.pack_into('<H', b, 0, i)
                struct.pack_into('<H', b, 2, j)
                for k in range(math.ceil(self.len_q_bytes/16)):
                    c = SHPRG.__aes128_16bytesonly(seedA, b)
                    A[i][j] += int.from_bytes(c, byteorder='big') * (2 ** (k*128) )
                    A[i][j] %= self.q
        return A

    
    def genRandom(self, s):
        s_np = np.array(s)
        # print(s_np)

        output = np.dot(self.A, s_np)
        # print(output)
        # print(type(output[0][0]))

        output = output* self.p 
        # print(output)
        # print(type(output[0][0]))

        output = output // self.q
        # print(output)
        # print(type(output[0][0]))
        # print(output* self.q )

        output = output % self.p
        # print(output)
        # print(type(output[0][0]))

        # print(type(self.q))
        # print(self.q)
        # output = np.floor(output) % self.p
        # output = self.__matrix_mul_round(self.A, s)
        return output

if __name__ == "__main__":
    n = 2 ** 1
    m = 2 ** 15
    EQ = 256
    EP = 128
    prg = SHPRG(input=n, output = m , EQ = EQ, EP = EP, load_A = "")
    s1 = [[i+1] for i in range(n)]
    s2 = [[78] for _ in range(n)]
    s3 = [[i+1+78] for i in range(n)]

    a = prg.genRandom(s1)
    b = prg.genRandom(s2)
    c = prg.genRandom(s3)
    # c = prg.genRandom((np.array(s1)+np.array(s2)).tolist())
    print(a)
    print(b)
    print(c)
    print(2**EP)
    for i in range(m):
        x = (a[i][0] + b[i][0] - c[i][0]) % 2**EP
        if(x != 2**EP-1 and x>1):
            print("error:")
            print(x)


    
    
    