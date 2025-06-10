# Convenience functions operating on string or list of strings.
# todo: func to remove punctuation
# todo: mirror str_util.R

import re
import string
import polars as pl
from phtrs import config as phon_config
from collections import Counter

punc = string.punctuation
smart_punc = '“”‘’'
punc = punc + smart_punc
punc_regexp = r"[" + re.escape(punc) + r"]"


def squish(word):
    """
    Collapse consecutive spaces, remove leading/trailing spaces.
    see: https://stringr.tidyverse.org/reference/str_trim.html
    """
    if isinstance(word, list):
        return [squish(wordi) for wordi in word]
    ret = re.sub('[ ]+', ' ', word)
    ret = ret.strip()
    return ret


def str_sep(word, syms=None, regexp=None):
    """
    Separate symbols in word with spaces.
    # see: torchtext.data.functional.simple_space_split
    """
    if syms is None and regexp is None:
        regexp = "(.)"
    if regexp is None:
        syms.sort(key=lambda x: len(x), reverse=True)
        regexp = '(' + '|'.join(syms) + ')'

    if isinstance(word, list):
        return [str_sep(wordi, syms, regexp, sep) for wordi in word]

    ret = re.sub(regexp, "\\1 ", word)
    ret = squish(ret)
    return ret


def add_delim(word, sep=False, edge='both'):
    """
    Add begin/end delimiters to space-separated string.
    """
    if isinstance(word, list):
        return [add_delim(wordi, sep, edge) for wordi in word]
    ret = word
    if sep:
        ret = ' '.join(ret)
    if edge == 'begin':
        ret = f'{phon_config.bos} {ret}'
    elif edge == 'end':
        ret = f'{ret} {phon_config.eos}'
    else:  # default edge == 'both'
        ret = f'{phon_config.bos} {ret} {phon_config.eos}'
    return ret


def remove_delim(word):
    """
    Remove begin/end delimiters.
    """
    if isinstance(word, list):
        return [remove_delim(wordi) for wordi in word]
    ret = word
    ret = re.sub(f'{phon_config.bos}', '', ret)
    ret = re.sub(f'{phon_config.eos}', '', ret)
    ret = squish(ret)
    return ret


def remove_syms(word, syms=None, regexp=None, sep=' '):
    """
    Remove designated symbols.
    """
    if syms is None and regexp is None:
        return word
    if regexp is None:
        regexp = '(' + '|'.join(syms) + ')'

    if isinstance(word, list):
        return [remove(wordi, syms, regexp, sep) for wordi in word]

    ret = re.sub(regexp, '', word)
    ret = squish(ret)
    return ret


def remove_punc(word):
    """
    Remove punctuation from word.
    """
    if isinstance(word, list):
        return [remove_punc(wordi) for wordi in word]
    ret = re.sub(punc_regexp, '', word)
    ret = squish(ret)
    return ret


def str_subs(word, subs={}, sep=' '):
    """
    Change transcription by applying dictionary of
    substitutions to string(s).
    """
    if isinstance(word, list):
        return [str_subs(wordi, subs, sep) for wordi in word]
    sep_flag = (sep is not None and sep != '')
    ret = word.split(sep) if sep_flag else word
    for s, r in subs.items():
        ret = [subs[x] if x in subs else x for x in ret]
    ret = sep.join(ret) if sep_flag else ''.join(ret)
    ret = squish(ret)
    return ret


retranscribe = str_subs  # Alias.

# def retranscribe_sep(x, subs, sep=' '):
#     """
#     Change transcription by applying dictionary of
#     substitutions to string(s) of separated segments.
#     """
#     if isinstance(x, list):
#         return [retranscribe_sep(xi, subs, sep) for xi in x]
#     y = x.split(sep)
#     y = [subs[yi] if yi in subs else yi for yi in y]
#     y = sep.join(y)
#     return y


def get_words(text, sep=' '):
    """
    Get words types and type frequencies from text(s).
    """
    words = Counter()
    if isinstance(text, pl.Series):
        text = text.to_list()
    if isinstance(text, list):
        for texti in text:
            words.update(texti.split(sep))
    else:
        words.update(text.split(sep))
    words = pl.DataFrame({ \
        'word': words.keys(),
        'freq': words.values() })
    words = words.sort(by='freq', descending=True)
    return words


def unigram_tokens(word, sep=' '):
    """
    Get unigram tokens from one word.
    """
    if sep is not None and sep != '':
        ret = word.split(sep)
    else:
        ret = word
    return ret


def unigrams(word, sep=' '):
    """
    Get unigram types and type frequencies from word(s).
    """
    ret = Counter()
    if isinstance(word, pl.Series):
        word = word.to_list()
    if isinstance(word, list):
        for wordi in word:
            ret.update(unigram_tokens(wordi, sep))
    else:
        ret.update(unigram_tokens(word, sep))
    ret = pl.DataFrame({ \
        'sym': ret.keys(),
        'freq': ret.values() })
    return ret


get_symbols = unigrams  # Alias.


def bigram_tokens(word, sep=' '):
    """
    Get bigram tokens from one word.
    """
    if sep is not None and sep != '':
        ret = word.split(sep)
    else:
        ret = word
    ret = list(zip(ret[:-1], ret[1:]))
    return ret


def bigrams(word, sep=' '):
    """
    Get bigrams and their type frequencies from word(s).
    """
    ret = Counter()
    if isinstance(word, pl.Series):
        word = word.to_list()
    if isinstance(word, list):
        for wordi in word:
            ret.update(bigram_tokens(wordi, sep))
    else:
        ret.update(bigram_tokens(word, sep))
    ret = pl.DataFrame({ \
        'bigram': ret.keys(),
        'freq': ret.values() })
    return ret


def lcp(x, y, prefix=True):
    """
    Longest common prefix (or suffix) of two segment sequences.
    """
    if x == y:
        return x
    if not prefix:
        x = x[::-1]
        y = y[::-1]
    n_x, n_y = len(x), len(y)
    n = max(n_x, n_y)
    for i in range(n + 1):
        if i >= n_x:
            match = x
            break
        if i >= n_y:
            match = y
            break
        if x[i] != y[i]:
            match = x[:i]
            break
    if not prefix:
        match = match[::-1]
    return match


def test():
    #phon_config.init({'epsilon': '<eps>', 'bos': '>', 'eos': '<'})
    print(phon_config.bos)
    print(phon_config.eos)
    print(squish(' t  e s  t   '))
    print(str_sep('cheek', syms=['ch', 'ee', 'k']))
    print(str_sep('cheek', regexp='(ch|ee|k)'))
    print(add_delim('test'))
    print(remove_syms('testing', syms='aeiou'))
    print(remove_punc('[(testing).!]?'))


if __name__ == "__main__":
    test()
