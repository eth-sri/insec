class UnicodeTokenizer():
    def __init__(self):
        self.vocab_size = 0x10FFFF

    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, id):
        return chr(id)

class AsciiTokenizer():
    def __init__(self):
        self.vocab_size = 128

    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, id):
        return chr(id)
