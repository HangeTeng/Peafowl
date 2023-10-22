import hashlib

class PRF:
    def __init__(self, secret_key, hash_function='md5'):
        self.secret_key = secret_key
        self.hash_function = hash_function

    def compute(self, message):
        # Use the specified hash function to compute the PRF value
        message_bytes = str(message).encode('utf-8')
        secret_key_bytes = self.secret_key.encode('utf-8')
        # Concatenate the message and secret_key
        data = message_bytes + secret_key_bytes

        # Compute the hash based on the selected hash_function
        if self.hash_function == 'sha256':
            hash_result = hashlib.sha256(data).digest()
        elif self.hash_function == 'md5':
            hash_result = hashlib.md5(data).digest()
        else:
            raise ValueError("Unsupported hash function")

        # Convert the hash to an integer
        prf_value = int.from_bytes(hash_result, byteorder='big')

        return prf_value

# Example usage with different hash functions:
if __name__ == "__main__":
    secret_key = "super_secret_key"
    
    # Using SHA-256
    prf_sha256 = PRF(secret_key, hash_function='sha256')
    message = "Hello, World!"
    prf_value_sha256 = prf_sha256.compute(message)
    print(f"Using SHA-256:")
    print(f"Message: {message}")
    print(f"PRF Value: {prf_value_sha256}")

    # Using MD5
    prf_md5 = PRF(secret_key, hash_function='md5')
    message = "Hello, World!"
    prf_value_md5 = prf_md5.compute(message)
    print("\nUsing MD5:")
    print(f"Message: {message}")
    print(f"PRF Value: {prf_value_md5}")
