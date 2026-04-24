"""
Contains the Regex Tokenizer class and a few common helper functions.
"""
try:
    from .base import get_stats, merge, render_token, replace_control_characters
    from .basic import BasicTokenizer
except ImportError:
    from base import get_stats, merge, render_token, replace_control_characters
    from basic import BasicTokenizer

import regex as re


class RegexTokenizer(BasicTokenizer):
    """A BPE tokenizer that uses regular expressions to find pairs to merge"""
    def __init__(self):
        super().__init__()
        
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
        self.pattern=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""



    def train(self, text, vocab_size, verbose=False):
        """
        Create the merges and the vocab.
        """
        pattern = re.compile(self.pattern)

        # split the text using the pattern
        text_chunks = re.findall(pattern, text)

        # convert each chunk to utf-8 bytes
        ids_chunks = []
        for tc in text_chunks:
            ids_chunks.append(list(map(int, tc.encode('utf-8'))))


        last_vocab = max(list(self.vocab.keys())) + 1

        merges_to_do = vocab_size - 256
        for i in range(0, merges_to_do):
            # get the stats for each chunk
            #stats_for_chunk = [get_stats(ids) for ids in ids_chunks]

            stats = {}
            #stats = self._merge_stats(stats_for_chunk)
            for ids in ids_chunks:
                stats = get_stats(ids, stats)

            top_pair = max(stats, key=stats.get)

            new_token = last_vocab + i

            ids_chunks = [merge(ids, top_pair, new_token) for ids in ids_chunks]

            self.merges[top_pair] = new_token

        # create new vocab
        self.vocab = self._build_vocab()

    def encode(self, text, verbose=False):
        # Tokenizer can encode a string into a list of integers

        pattern = re.compile(self.pattern)

        text_chunks = re.findall(self.pattern, text)

        tokens_chunks = [list(map(int, t.encode('utf-8'))) for t in text_chunks]
        
        tokens_enc = []
        for tt in tokens_chunks:
            for pair_to_merge, new_token in self.merges.items():
                tt = merge(tt, pair_to_merge, new_token)
            tokens_enc.append(tt)

        ids = [t for ts in tokens_enc for t in ts]

        return ids
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        original_str = b"".join([self.vocab[idx] for idx in ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str
    

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens

        # create new pattern
        for special_token in special_tokens.keys():
            special_token = re.escape(special_token) # escape special characters in the token
            self.pattern = rf"{special_token}|" + self.pattern


    @staticmethod
    def _merge_stats(stats_list):
        """
        Combine a list of dict with pair frequency in a single dict
        """
        full_stats = {}
        for s in stats_list:
            for k,v in s.items():
                if k not in full_stats.keys():
                    full_stats[k] = v
                else:
                    current_value = full_stats.get(k)
                    full_stats[k] = current_value + v
                    del current_value
        return full_stats

if __name__ == "__main__":

    import os


    tokenizer = RegexTokenizer()

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

    tokens = tokenizer.encode("Hello world!!!!?", verbose=False)

    print(f"decoded: {tokenizer.decode(tokens)}")
    tokens = tokenizer.encode("H")
    print(tokens)

    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_tokenizers')

    tokenizer.save("regex_tokenizer")