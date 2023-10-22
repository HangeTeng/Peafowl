import numpy as np

class FixedPointEncoder:
    def __init__(self, precision_bits=16):
        self.precision_bits = precision_bits
        self.scale = 2 ** precision_bits

    def encode(self, float_array):
        if not isinstance(float_array, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        int_array = (float_array * self.scale).astype(np.int64)
        return int_array

    def decode(self, int_array):
        if not isinstance(int_array, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        float_array = int_array / self.scale
        return float_array

def mod_range(a, p):
    "convert to  [-p/2, p/2 - 1]"
    r = (a % p) - p // 2
    return r

if __name__ == "__main__":
    encoder = FixedPointEncoder(precision_bits=16)

    float_values = np.array([1, 2, 3])
    encoded_values = encoder.encode(float_values)
    decoded_values = encoder.decode(encoded_values)

    print("Original float values:", float_values)
    print("Encoded values:", encoded_values)
    print("Decoded float values:", decoded_values)