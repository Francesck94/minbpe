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
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.shuffle=shuffle
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.verbose = verbose
        #self.merges = self._shuffle_merges()
        self.merges = gpt4_merges
        self.vocab = self._build_vocab_shuffled() # int -> bytes


    # def _shuffle_merges(self):
    #     # shuffle the merges according to the GPT4 byte shuffle, so that we can use the same merges
    #     new_merges = {}
    #     for pair, new_token in self.merges.items():
    #         new_pair = (self.byte_shuffle.get(pair[0], pair[0]), self.byte_shuffle.get(pair[1], pair[1]))
    #         new_merges[new_pair] = new_token
    #     return new_merges
    
    
    def _build_vocab_shuffled(self):
        # vocab is simply and deterministically derived from merges
        vocab_orig = {idx: bytes([idx]) for idx in range(256)}
        vocab = {idx: vocab_orig[self.byte_shuffle.get(idx, idx)] for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            #p0, p1 = (self.byte_shuffle.get(p0, p0), self.byte_shuffle.get(p1, p1))
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
        text_chunks = re.findall(self.pattern, text)

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

    def merge_gpt(self,ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                if self.verbose:
                    print(f"Merging pair {pair} into new token {idx}")
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text, verbose=True):
        # Tokenizer can encode a string into a list of integers

        pattern = re.compile(self.pattern)

        text_chunks = re.findall(pattern, text)

        tokens_chunks = [list(map(int, t.encode('utf-8'))) for t in text_chunks]
        print(f"Initial tokens chunks: {tokens_chunks}")

        # shuffle the bytes according to the GPT4 byte shuffle, so that we can use the same merges
        if self.shuffle:
            tokens_chunks =  [
                list(map(self._apply_shuffle, tc))
                for tc in tokens_chunks
                ]
        print(f"Tokens chunks after shuffle: {tokens_chunks}")
        
        tokens_enc = []

        parallel = False
        if parallel:
            tokens_enc = list(map(lambda x: self._encode_chunk(x[1], index=x[0], total_chunks=len(tokens_chunks)), enumerate(tokens_chunks)))
        else:
            for idx, tt in enumerate(tokens_chunks):
                if verbose:
                    if idx % 100 == 0:
                        print(f"Encoding chunk {idx}/{len(tokens_chunks)}")
                for pair_to_merge, new_token in self.merges.items():
                    tt = self.merge_gpt(tt, pair_to_merge, new_token)
                tokens_enc.append(tt)

        ids = [t for ts in tokens_enc for t in ts]

        return ids
    
    def _encode_chunk(self, tok_chunk, index=None, total_chunks=None):
        """Helper function for encode, to encode a single chunk."""
        if index is not None and total_chunks is not None:
            if index % 100 == 0:
                if self.verbose:
                    print(f"Encoding chunk {index}/{total_chunks}")
        for pair_to_merge, new_token in self.merges.items():
            tok_chunk = self.merge_gpt(tok_chunk, pair_to_merge, new_token)
        return tok_chunk
    
    # def _merge_multithreaded(self, tokens_chunks):
    #     # merge the tokens using multiple threads. This is a helper function for the encode method.

    #     def _merge_single_chunk(tokens, idx, total_chunks):
    #         if idx % 100 == 0:
    #             print(f"Encoding chunk {idx}/{total_chunks}")
    #         for pair_to_merge, new_token in self.merges.items():
    #             tokens = merge(tokens, pair_to_merge, new_token)
    #         return tokens
    #     # we can use the concurrent.futures module to do this.
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = []
    #         for idx, tt in enumerate(tokens_chunks):
    #             futures.append(executor.submit(_merge_single_chunk, tt, idx, len(tokens_chunks)))
    #         results = [f.result() for f in futures]
    #     return results
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        if self.shuffle:
            ids = [self._apply_unshuffle(t) for t in ids]
        original_str = b"".join([self.vocab[idx] for idx in ids])
        original_str = original_str.decode("utf-8", errors="replace")
        return original_str
    

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens

        # create new pattern
        for special_token in special_tokens.keys():
            special_token = re.escape(special_token) # escape special characters in the token
            self.pattern = rf"{special_token}|" + self.pattern

    # TODO: FIX THIS
    def _apply_shuffle(self, token):
        new_token = self.byte_shuffle[token] if token in self.byte_shuffle else token
        return new_token
    
    def _apply_unshuffle(self, token):
        # reverse the byte shuffle
        unshuffled = {v: k for k, v in self.byte_shuffle.items()}
        new_token = unshuffled[token] if token in unshuffled else token
        return new_token


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
    text = "a"
    print(f"Original text: {text}")
    tokens = tokenizer.encode(text, verbose=True)
    tokens_gpt_encoded = enc_gpt.encode(text)
    print(f"GPT4 encoded tokens: {tokens_gpt_encoded}")
    print(f"Encoded tokens: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded}")
    print(f"is decoded text same as original? {decoded == text}")
    

    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_tokenizers')

    tokenizer.save("gpt4_tokenizer")