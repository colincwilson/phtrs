# Convenience functions operating on string or list of strings.

import re
import polars as pl
from phtrs import config as phon_config
from collections import Counter


def squish(x):
    """
    Collapse consecutive spaces, remove leading/trailing spaces.
    see: https://stringr.tidyverse.org/reference/str_trim.html
    """
    if isinstance(x, list):
        return [squish(xi) for xi in x]
    y = re.sub('[ ]+', ' ', x)
    return y.strip()


def sep_chars(x):
    """
    Separate characters with space.
    """
    # see: torchtext.data.functional.simple_space_split
    if isinstance(x, list):
        return [sep_chars(xi) for xi in x]
    return ' '.join(x)


def add_delim(x, sep=False, edge='both'):
    """
    Add begin/end delimiters to space-separated string.
    """
    if isinstance(x, list):
        return [add_delim(xi, sep) for xi in x]
    if sep:
        x = ' '.join(x)
    if edge == 'begin':
        y = f'{phon_config.bos} {x}'
    elif edge == 'end':
        y = f'{x} {phon_config.eos}'
    else:  # default edge == 'both'
        y = f'{phon_config.bos} {x} {phon_config.eos}'
    return y


def remove_delim(x):
    """
    Remove begin/end delimiters.
    """
    if isinstance(x, list):
        return [remove_delim(xi) for xi in x]
    y = re.sub(f'{phon_config.bos}', '', x)
    y = re.sub(f'{phon_config.eos}', '', y)
    return squish(y)


def remove(x, syms):
    """
    Remove designated symbols.
    """
    if isinstance(x, list):
        return [remove(xi, syms) for xi in x]
    y = x
    for sym in syms:
        y = re.sub(sym, '', y)
    return squish(y)


def retranscribe(x, subs):
    """
    Change transcription by applying dictionary of
    substitutions to string(s).
    """
    if isinstance(x, list):
        return [retranscribe(xi, subs) for xi in x]
    y = x
    for s, r in subs.items():
        y = re.sub(s, r, y)
    return squish(y)


def retranscribe_sep(x, subs, sep=' '):
    """
    Change transcription by applying dictionary of
    substitutions to string(s) of separated segments.
    """
    if isinstance(x, list):
        return [retranscribe_sep(xi, subs, sep) for xi in x]
    y = x.split(sep)
    y = [subs[yi] if yi in subs else yi for yi in y]
    y = sep.join(y)
    return y


def get_words(text, sep=' '):
    """
    Get words with frequencies from text(s).
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
    return words


def get_symbols(word, sep=' '):
    """
    Get symbols with frequencies from word(s).
    """
    syms = Counter()
    if isinstance(word, pl.Series):
        word = word.to_list()
    if isinstance(word, list):
        for wordi in word:
            if sep != '':
                wordi = wordi.split(sep)
            syms.update(wordi)
    else:
        syms.update(word.split(sep))
    syms = pl.DataFrame({ \
        'sym': syms.keys(),
        'freq': syms.values() })
    return syms


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
    phono_config.init({'epsilon': '<eps>', 'bos': '>', 'eos': '<'})
    print(phon_config.bos)
    print(phon_config.eos)


if __name__ == "__main__":
    test()
