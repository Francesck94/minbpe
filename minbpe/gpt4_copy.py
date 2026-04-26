"""
Contains the GPT4 Tokenizer class and a few common helper functions.
"""
try:
    from .base import get_stats, merge, render_token, replace_control_characters
    from .regex_tok import RegexTokenizer
except ImportError:
    from base import get_stats, merge, render_token, replace_control_characters
    from regex_tok import RegexTokenizer

from typing import Optional
import regex as re
import tiktoken
import concurrent.futures


def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

gpt4_merges = recover_merges(tiktoken.get_encoding("cl100k_base")._mergeable_ranks)

class GPT4Tokenizer(RegexTokenizer):
    """A BPE tokenizer that uses regular expressions to find pairs to merge"""
    def __init__(self, shuffle=False, verbose=False):
        super().__init__()
        enc = tiktoken.get_encoding("cl100k_base")
        #self.merges = {} # (int, int) -> int
        self.special_tokens_gpt4 = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
        #self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.special_tokens = self.special_tokens_gpt4

        self.shuffle=shuffle
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.inv_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        self.verbose = verbose
        #self.merges = self._shuffle_merges()
        self.merges = gpt4_merges
        self.vocab = self._build_vocab_gpt() # int -> bytes

    def _build_vocab_gpt(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            p0 = self.inv_byte_shuffle.get(p0, p0)
            p1 = self.inv_byte_shuffle.get(p1, p1)
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    
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

    def encode(self, text, allowed_special=None, verbose=False):
        # Tokenizer can encode a string into a list of integers

        pattern = re.compile(self.pattern)

        if allowed_special is not None:
            if allowed_special == "all":
                print("Registering GPT4 special tokens")
                #self.special_tokens = self.special_tokens_gpt4
                special_alt = "|".join(re.escape(s) for s in sorted(self.special_tokens.keys(), key=len, reverse=True))
                self.pattern = rf"(?:{special_alt})|{self.pattern}"
                print(f"New pattern: {self.pattern}")
            else:
                pass # TODO implement other options for allowed_special

        text_chunks = re.findall(self.pattern, text)

        if allowed_special is None:
            tokens_chunks = [list(map(int, t.encode('utf-8'))) for t in text_chunks]
        else:
            tokens_chunks = []
            for t in text_chunks:
                if t in self.special_tokens.keys():
                    tokens_chunks.append([self.special_tokens[t]])
                else:
                    tokens_chunks.append(list(map(int, t.encode('utf-8'))))

        tokens_chunks_shuffled = []
        for t_chunk in tokens_chunks:
            shuffled_chunk = []
            for tok in t_chunk:
                shuffled_chunk.append(self.byte_shuffle.get(tok, tok))
            tokens_chunks_shuffled.append(shuffled_chunk)

        
        tokens_enc = []
        for tt in tokens_chunks_shuffled:
            for pair_to_merge, new_token in self.merges.items():
                tt = merge(tt, pair_to_merge, new_token)
            tokens_enc.append(tt)

        ids = [t for ts in tokens_enc for t in ts]

        return ids
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        # unshuffle the bytes
        unshuffled_ids = []
        for idx in ids:
            unshuffled_ids.append(self.inv_byte_shuffle.get(idx, idx))
        original_str = b"".join([self.vocab[idx] for idx in unshuffled_ids])
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


    tokenizer = GPT4Tokenizer(shuffle=False, verbose=True)

    #print(vars(tokenizer))

    print(len(tokenizer.vocab))

    for idx in range(0, 256 + 10):
        print(f"{idx}: {tokenizer.vocab[idx].decode('utf-8', errors='replace')}")
    #print(tokenizer.vocab)

    # module path

    #filepath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'taylorswift.txt')
    #print(f"filepath: {filepath}")
    #with open(filepath, 'r') as f:
    #     text = f.read()

    # tokenizer.train(text, vocab_size=512, verbose=False)

    # #print(f"final vocab: {tokenizer.vocab}")

    # print(len(tokenizer.vocab))
    # #print(f"final vocab: {tokenizer.vocab}")
    # #print(f"\nfinal merges: {tokenizer.merges}\n")
    enc_gpt = tiktoken.get_encoding("cl100k_base")

    #text = "hello world!!!? (안녕하세요!) lol123 😉"
    #text = "<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM"
    text = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
    #text = "a"
    print(f"Original text: {text}")
    tokens = tokenizer.encode(text, allowed_special="all", verbose=True)
    tokens_gpt_encoded = enc_gpt.encode(text, allowed_special="all")
    print(f"GPT4 encoded tokens: {tokens_gpt_encoded}")
    print(f"Encoded tokens: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded}")
    print(f"is decoded text same as original? {decoded == text}")
    

    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_tokenizers')

    tokenizer.save("gpt4_tokenizer")