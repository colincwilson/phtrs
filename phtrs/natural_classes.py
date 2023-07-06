# -*- coding: utf-8 -*-

import numpy as np
from phtrs import config as phon_config


def subsumes(ftrs1, ftrs2):
    """
    Dictionary ftrs1 subsumes dictionary ftrs2 iff 
    every non-zero feature spec in ftrs1 is also in ftrs2.
    """
    for ftr, val1 in ftrs1.items():
        if val1 == 0 or val1 == '0':
            continue
        if ftr not in ftrs2:
            return False
        val2 = ftrs2[ftr]
        if val2 != val1:
            return False
    return True


def match(form, pattern, focus_idx=0):
    """
    Mark every symbol in space-separated form as 
    matching (1) or not matching (0) feature pattern 
    (sequence of feature-value dicts) with specified 
    focus index. Slow! (quadratic)
    """
    fm = phon_config.feature_matrix
    form = form.split(' ')
    m = len(pattern)
    n = len(form)
    ret = np.array([0.0] * n)

    for i in range(n - m + 1):
        flag = True
        for j in range(m):
            ftrs = fm.sym2ftrs[form[i + j]]
            if not subsumes(pattern[j], ftrs):
                flag = False
                break
        if flag:
            ret[i + focus_idx] = 1.0

    return ret
