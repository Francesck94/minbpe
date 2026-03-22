"""
Contains the Basic Tokenizer class and a few common helper functions.
"""
try:
    from .base import get_stats, merge, render_token, replace_control_characters
    from .base import Tokenizer
except ImportError:
    from base import get_stats, merge, render_token, replace_control_characters
    from base import Tokenizer



class BasicTokenizer(Tokenizer):
    """A basic implementation of the BPE tokenizer"""
    def __init__(self):
        super().__init__()
        
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode('utf-8')
        tokens = [int(t) for t in tokens]

        num_merges = vocab_size - 256
        new_token = max(self.vocab.keys()) + 1

        for _ in range(num_merges):
            stats = get_stats(tokens)

            top_pair = max(stats, key=stats.get)

            tokens = merge(tokens, top_pair, new_token)

            self.merges[top_pair] = new_token

            new_token += 1


        self.vocab = self._build_vocab()



    def encode(self, text, verbose=False):
        # Tokenizer can encode a string into a list of integers
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))

        #print("original tokens: ", tokens)
        #print(f"len ")

        for pair_to_merge, new_token in self.merges.items():
            if verbose:
                print("pair to merge: ", pair_to_merge)
            tokens = merge(tokens, pair_to_merge, new_token)

            if verbose:
                print("new tokens: ", tokens)

        return tokens


    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        original_str = b"".join([self.vocab[idx] for idx in ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str

if __name__ == "__main__":

    import os


    tokenizer = BasicTokenizer()

    #print(vars(tokenizer))

    print(len(tokenizer.vocab))

    # module path

    filepath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'taylorswift.txt')
    print(f"filepath: {filepath}")
    with open(filepath, 'r') as f:
        text = f.read()

    tokenizer.train(text, vocab_size=512, verbose=False)

    #print(f"final vocab: {tokenizer.vocab}")

    print(len(tokenizer.vocab))
    #print(f"final vocab: {tokenizer.vocab}")
    #print(f"\nfinal merges: {tokenizer.merges}\n")

    tokens = tokenizer.encode("Hello world", verbose=False)

    print(f"decoded: {tokenizer.decode(tokens)}")
    tokens = tokenizer.encode("H")
    print(tokens)

    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_tokenizers')

    tokenizer.save("basic_tokenizer")