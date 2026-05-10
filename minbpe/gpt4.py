"""
Contains the GPT4 Tokenizer class and a few common helper functions.
"""
try:
    from .base import get_stats, merge
    from .regex_tok import RegexTokenizer
except ImportError:
    from base import get_stats, merge
    from regex_tok import RegexTokenizer

from typing import Optional
import regex as re
import tiktoken
#from utils import timing


### utility functions for recovering merges from tiktoken's mergeable ranks
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

#gpt4_merges = recover_merges(tiktoken.get_encoding("cl100k_base")._mergeable_ranks)

class GPT4Tokenizer(RegexTokenizer):
    """A BPE tokenizer that uses regular expressions to find pairs to merge"""
    def __init__(self, verbose=False):
        super().__init__()
        enc = tiktoken.get_encoding("cl100k_base")
        self.special_tokens_gpt4 = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
        #self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.special_tokens = self.special_tokens_gpt4

        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.inv_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        self.verbose = verbose
        #self.merges = self._shuffle_merges()
        self.merges = recover_merges(tiktoken.get_encoding("cl100k_base")._mergeable_ranks)
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
        # inherit the pattern from the RegexTokenizer
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
            for ids in ids_chunks:
                stats = get_stats(ids, stats)

            top_pair = max(stats, key=stats.get)

            new_token = last_vocab + i

            ids_chunks = [merge(ids, top_pair, new_token) for ids in ids_chunks]

            self.merges[top_pair] = new_token

        # create new vocab
        self.vocab = self._build_vocab()

    def encode(self, text, allowed_special=None):
        # Tokenizer can encode a string into a list of integers

        text_chunks = []

        special_tokens = {}

        # split the text into segments that are either special tokens or not, and filter out empty segments
        segments, special_tokens = self._handle_special_tokens_in_encode(text, allowed_special)
        
        
        for s in segments:
            if s in special_tokens.keys():
                text_chunks.append(s)
            else:
                text_chunks.extend(re.findall(self.pattern, s))
        
        if self.verbose:
            print(f"text_chunks: {text_chunks}")

        # convert each chunk to utf-8 bytes
        if allowed_special is None:
            tokens_chunks = [list(map(int, t.encode('utf-8'))) for t in text_chunks]
        else:
            # if allowed_special is not None, we have to be careful to not encode the special tokens as utf-8 bytes, but rather use their assigned integer ids. So we check if each chunk is a special token, and if so we use the assigned integer id, otherwise we encode as utf-8 bytes.
            tokens_chunks = []
            for t in text_chunks:
                if t in special_tokens.keys():
                    tokens_chunks.append([special_tokens[t]])
                else:
                    tokens_chunks.append(list(map(int, t.encode('utf-8'))))


        tokens_chunks_shuffled = self._apply_shuffle_in_encode(tokens_chunks)

        # apply the merges to each chunk
        if self.verbose:
            print(f"applying merges...")

        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1]) # sort by the new token id, which is the value in the merges dict

        tokens_enc = map(lambda tok_chunk: self._apply_merges_on_chunk(sorted_merges, tok_chunk), tokens_chunks_shuffled)

        ids = [t for ts in tokens_enc for t in ts]

        return ids
    
    
    def _apply_merges_on_chunk(self, sorted_merges, tok_chunk):
        found_merge = True
        tok_chunk_pair = list(zip(tok_chunk[:-1], tok_chunk[1:]))
        while found_merge:
            found_merge = False
            for pair_to_merge, new_token in sorted_merges:
                if pair_to_merge in tok_chunk_pair:
                    # if the pair to merge is in the token list, then merge it
                    tok_chunk = merge(tok_chunk, pair_to_merge, new_token)
                    tok_chunk_pair = list(zip(tok_chunk[:-1], tok_chunk[1:]))
                    found_merge = True
                    break # we have to break here because the merges have to be applied sequentially, and we can't apply multiple merges at the same time because they might interfere with each other. For example, if we have merges (a, b) -> x and (x, c) -> y, and our token chunk is [a, b, c], we have to first merge (a, b) to get [x, c], and then merge (x, c) to get [y]. If we tried to apply both merges at the same time, we would not know whether to merge (a, b) or (x, c) first.
        return tok_chunk

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string

        # unshuffle the bytes
        unshuffled_ids = []
        for idx in ids:
            unshuffled_ids.append(self.inv_byte_shuffle.get(idx, idx))
        original_str = b"".join([self.vocab[idx] for idx in unshuffled_ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str
    

    def _create_special_pattern(self, special_tokens):
        # create a regex pattern that matches any of the special tokens, sorted by length
        special_pattern = "|".join("("+re.escape(s)+")" for s in sorted(special_tokens.keys(), key=len, reverse=True))
        return special_pattern
    
    def _handle_special_tokens_in_encode(self, text, allowed_special):
        # if allowed_special is "all", we want to treat all special tokens as indivisible units
        # if allowed_special is a list of special tokens, we want to treat those as indivisible units
        # otherwise, we don't treat any special tokens as indivisible units

        special_tokens = {}
        if allowed_special == "all":
            special_tokens = self.special_tokens
        elif isinstance(allowed_special, list):
            special_tokens = {s: self.special_tokens[s] for s in allowed_special}
        else:
            return [text], special_tokens

        # split the text into segments that are either special tokens or not, and filter out empty segments
        special_pattern = self._create_special_pattern(special_tokens)
        segments = re.split(special_pattern, text, ignore_unused=True)
        segments = [s for s in segments if s is not None and s != '']
        return segments, special_tokens
    
    #@timing
    def _apply_shuffle_in_encode(self, tokens_chunks):
        # apply the byte shuffle to the tokens
        tokens_chunks_shuffled = []
        for t_chunk in tokens_chunks:
            shuffled_chunk = []
            for tok in t_chunk:
                shuffled_chunk.append(self.byte_shuffle.get(tok, tok))
            tokens_chunks_shuffled.append(shuffled_chunk)
        return tokens_chunks_shuffled



if __name__ == "__main__":

    import os


    tokenizer = GPT4Tokenizer(verbose=False)

    #print(vars(tokenizer))

    print(len(tokenizer.vocab))

    #for idx in range(0, 256 + 10):
    #     print(f"{idx}: {tokenizer.vocab[idx].decode('utf-8', errors='replace')}")
    #print(tokenizer.vocab)

    # module path

    filepath = os.path.join(os.path.dirname(__file__), '..', 'tests', 'taylorswift.txt')
    print(f"filepath: {filepath}")
    with open(filepath, 'r') as f:
        text = f.read()

    text = text[:1000]
    # #print(f"final vocab: {tokenizer.vocab}")

    
    enc_gpt = tiktoken.get_encoding("cl100k_base")

    #text = "hello world!!!? (안녕하세요!) lol123 😉"
    #text = "<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM"
#     text = """
# <|endoftext|>Hello world this is one document
# <|endoftext|>And this is another document
# <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
# <|endoftext|>Last document!!! 👋<|endofprompt|>
# """.strip()
    #text = "a"
    #print(f"Original text: {text}")
    import time
    start = time.perf_counter()
    tokens = tokenizer.encode(text, allowed_special="all")
    elapsed = time.perf_counter() - start
    print(f"Encoding took {elapsed:.4f}s")

    tokens_gpt_encoded = enc_gpt.encode(text, allowed_special="all")
    elapsed = time.perf_counter() - start
    #print(f"TikToken tokenizer: {tokens_gpt_encoded}")
    #print(f"Custom tokenizer: {tokens}")
    decoded = tokenizer.decode(tokens)
   # print(f"Decoded text: {decoded}")
    print(f"is decoded text same as original? {decoded == text}")
    

    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_tokenizers')

    tokenizer.save("gpt4_tokenizer")