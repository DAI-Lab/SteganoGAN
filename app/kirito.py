import zlib
from reedsolo import RSCodec, ReedSolomonError

def bytearray_to_bits(x):
    # Convert bytearray to a list of bits
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_bytearray(bits):
    # Convert a list of bits to a bytearray
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b*8:(b+1)*8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    return bytearray(ints)

class KiritoCodec(object):
    
    def __init__(self):
        self.rs = RSCodec(250)
    
    def encode(self, text):
        assert type(text) == str, "expected a string"
        x = zlib.compress(text.encode("utf-8"))
        x = self.rs.encode(bytearray(x))
        return x
    
    def decode(self, x):
        assert type(x) == bytearray, "expected a bytearray"
        try:
            text = self.rs.decode(x)
            text = zlib.decompress(text)
            return text.decode("utf-8")
        except:
            return False

kc = KiritoCodec()

def to_bytearray(s):
    # Convert string to a bytearray
    return kc.encode(s)

def from_bytearray(bits):
    # Convert a bytearray to a string
    return kc.decode(bits)
