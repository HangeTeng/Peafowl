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
    def __init__(self, input=2, output =4, EQ = 64, EP = 32, seedA = bytes(0x355678), load_A=''):
        """To construct a shprg with inputsize = input*EQ bit and outputsize = output * EP bit"""
        self.gen = self.__genAES128
        self.shake = self.__shake128
        # setParams
        self.n = input
        self.m = output
        self.q = 2 ** EQ
        self.p = 2 ** EP
        self.len_seedAES = 128
        self.len_q_bytes = int(EQ / 8)
        self.len_p_bytes = int(EP / 8)
        self.len_seedAES_bytes = int(self.len_seedAES / 8)

        if load_A == '':
            self.A = self.gen(self.shake(seedA, self.len_seedAES_bytes))
            self.A = np.array(self.A)
            np.save('A_n{}_m{}_q{}_p{}.npy'.format(input,output,EQ,EP),self.A)
        else: 
            self.A = np.load(load_A, allow_pickle=True)
    
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
    
    # @timer
    def __genAES128(self, seedA):
        """Generate matrix A using AES-128 (SHPRG specification, Algorithm 7)"""
        A = [[0 for _ in range(self.n)] for _ in range(self.m)]
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
    
    @staticmethod
    def genMatrixAES128(seed = bytes(0x355678), n = 2, m = 4, EQ = 64):
        """Generate matrix A using AES-128 (SHPRG specification, Algorithm 7)"""
        q = 2 ** EQ
        len_q_bytes = int(EQ / 8)
        A = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                b = bytearray(16)
                struct.pack_into('<H', b, 0, i)
                struct.pack_into('<H', b, 2, j)
                for k in range(math.ceil(len_q_bytes/16)):
                    c = SHPRG.__aes128_16bytesonly(seed, b)
                    A[i][j] += int.from_bytes(c, byteorder='big') * (2 ** (k*128) )
                    A[i][j] %= q
        return A

    # @timer
    def genRandom(self, s_np):
        output = np.dot(self.A, s_np)
        output = output* self.p 
        output = output // self.q
        output = output % self.p
        return output

if __name__ == "__main__":
    n = 2 ** 8
    m = 2 ** 15
    EQ = 64
    EP = 32
    # load_A = 'A_{}_{}.npy'.format(n,m)
    load_A = ""
    prg = SHPRG(input=n, output = m , EQ = EQ, EP = EP, load_A = load_A)
    s1 = np.array([[i+1] for i in range(n)])
    s2 = np.array([[78] for _ in range(n)])
    s3 = np.array([[i+1+78] for i in range(n)])

    a = prg.genRandom(s1)
    b = prg.genRandom(s2)
    c = prg.genRandom(s3)
    for i in range(m):
        x = (a[i][0] + b[i][0] - c[i][0]) % 2**EP
        if(x != 2**EP-1 and x>1):
            print("error!")


    
    
    