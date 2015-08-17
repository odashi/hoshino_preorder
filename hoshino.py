#!/usr/bin/python3

import sys
from binascii import crc32
from nltk.tree import Tree
from copy import deepcopy
from collections import defaultdict


def myhash(text):
    bits = 28
    mask = (1 << bits) - 1
    while True:
        ret = crc32(text.encode()) & mask
        if ret > 0:
            return ret
        text += ' '


def binarize(tree):
    if not isinstance(tree, Tree):
        return tree
    
    children = [binarize(ch) for ch in tree]
    while len(children) > 2:
        temp = Tree('(' + tree.label() + 'bar)')
        temp.append(children[-2])
        temp.append(children[-1])
        children = children[:-2] + [temp]
    
    ret = Tree('(' + tree.label() + ')')
    for ch in children:
        ret.append(ch)
    
    return ret


def add_span(tree):
    def recursive(tree, begin):
        if not isinstance(tree, Tree):
            return 1
        end = begin
        for ch in tree:
            end += recursive(ch, end)
        tree.set_label((tree.label(), begin, end))
        return end - begin

    recursive(tree, 0)
    return tree


def read_align(text, num_words):
    ret = [-1] * num_words
    for la in text.split():
        trg, src = tuple(la.split('-'))
        ret[int(src)] = int(trg)
    
    return ret


def kendall(align):
    ret = 0
    for i in range(len(align) - 1):
        for j in range(i+1, len(align)):
            if align[i] >= 0 and align[j] >= 0 and align[i] < align[j]:
                ret += 1
    return ret


def make_reorder(tree, align):
    if not isinstance(tree, Tree):
        return []
    if len(tree) != 2:
        return make_reorder(tree[0], align)

    ret = make_reorder(tree[0], align) + make_reorder(tree[1], align)

    name = tree.label()[0]
    if name not in {'動詞p', '名詞p'}:
        return ret

    l_span = tree[0].label()[1:3]
    r_span = tree[1].label()[1:3]
    l_align = align[l_span[0]:l_span[1]]
    r_align = align[r_span[0]:r_span[1]]
    m_score = kendall(l_align + r_align)
    w_score = kendall(r_align + l_align)

    label = 0
    if m_score > w_score: label = 1
    elif m_score < w_score: label = -1
    #else: label = 0
    #print(label, l_span[0], l_span[1], r_span[1], l_align, r_align)
    
    ret.append((label, l_span[0], l_span[1], r_span[1]))

    return ret


def convert_tree(tree, reorder):
    if not isinstance(tree, Tree):
        return
    elif len(tree) != 2:
        convert_tree(tree[0], reorder)
    else:        
        _, l, r = tree.label()
        for label, a, _, b in reorder:
            if label == -1 and l == a and r == b:
                tree[0], tree[1] = tree[1], tree[0]
        for ch in tree:
            convert_tree(ch, reorder)


def convert_align(align, reorder):
    for label, a, b, c in reorder:
        if label == -1:
            align = align[:a] + align[b:c] + align[a:b] + align[c:]
    return align


def extract_words(tree):
    if not isinstance(tree, Tree):
        return [tree]
    ret = []
    for ch in tree:
        ret += extract_words(ch)
    return ret


def extract_pos(tree):
    if isinstance(tree, Tree) and len(tree) == 1 and not isinstance(tree[0], Tree):
        return [tree.label()[0]]
    ret = []
    for ch in tree:
        ret += extract_pos(ch)
    return ret


def make_align_str(align):
    ret = ''
    for a, b in enumerate(align):
        if b >= 0:
            ret += '%d-%d ' % (b,a)
    return ret.strip()


def make_features(tree, words, pos, left, mid, right):
    while True:
        if len(tree) == 1:
            tree = tree[0]
        else:
            _, l, r = tree.label()
            if l == left and r == right:
                break

            if right <= tree[0].label()[2]:
                tree = tree[0]
            else:
                tree = tree[1]

    CONTEXT = 2
    wi_list = words[left:mid]
    wil_list = wi_list[:CONTEXT] if len(wi_list) > CONTEXT else wi_list
    wir_list = wi_list[-CONTEXT:] if len(wi_list) > CONTEXT else wi_list
    wj_list = words[mid:right]
    wjl_list = wj_list[:CONTEXT] if len(wj_list) > CONTEXT else wj_list
    wjr_list = wj_list[-CONTEXT:] if len(wj_list) > CONTEXT else wj_list
    ti_list = pos[left:mid]
    til_list = ti_list[:CONTEXT] if len(ti_list) > CONTEXT else ti_list
    tir_list = ti_list[-CONTEXT:] if len(ti_list) > CONTEXT else ti_list
    tj_list = pos[mid:right]
    tjl_list = tj_list[:CONTEXT] if len(tj_list) > CONTEXT else tj_list
    tjr_list = tj_list[-CONTEXT:] if len(tj_list) > CONTEXT else tj_list

    np_str = ':NP=' + tree.label()[0]
    nl_str = ':NL=' + tree[0].label()[0]
    nr_str = ':NR=' + tree[1].label()[0]

    wi_str = ':WI=' + '_'.join(wi_list)
    wil_str = ':WIL=' + '_'.join(wil_list)
    wir_str = ':WIR=' + '_'.join(wir_list)
    wj_str = ':WJ=' + '_'.join(wj_list)
    wjl_str = ':WJL=' + '_'.join(wjl_list)
    wjr_str = ':WJR=' + '_'.join(wjr_list)
    ti_str = ':TI=' + '_'.join(ti_list)
    til_str = ':TIL=' + '_'.join(til_list)
    tir_str = ':TIR=' + '_'.join(tir_list)
    tj_str = ':TJ=' + '_'.join(tj_list)
    tjl_str = ':TJL=' + '_'.join(tjl_list)
    tjr_str = ':TJR=' + '_'.join(tjr_list)

    features = []

    features.append(np_str)
    features.append(nl_str)
    features.append(nr_str)
    features.append(np_str + nl_str)
    features.append(np_str + nr_str)
    features.append(nl_str + nr_str)
    features.append(np_str + nl_str + nr_str)

    features.append(wi_str)
    features.append(wil_str)
    features.append(wir_str)
    features.append(wj_str)
    features.append(wjl_str)
    features.append(wjr_str)
    features.append(ti_str)
    features.append(til_str)
    features.append(tir_str)
    features.append(tj_str)
    features.append(tjl_str)
    features.append(tjr_str)

    features.append(wi_str + wj_str)
    features.append(ti_str + tj_str)
    features.append(wi_str + wj_str + ti_str + tj_str)
    features.append(wil_str + wjl_str)
    features.append(til_str + tjl_str)
    features.append(wil_str + wjl_str + til_str + tjl_str)
    features.append(wir_str + wjl_str)
    features.append(tir_str + tjl_str)
    features.append(wir_str + wjl_str + tir_str + tjl_str)
    features.append(wil_str + wjr_str)
    features.append(til_str + tjr_str)
    features.append(wil_str + wjr_str + til_str + tjr_str)
    features.append(wir_str + wjr_str)
    features.append(tir_str + tjr_str)
    features.append(wir_str + wjr_str + tir_str + tjr_str)

    #return features

    hashed = defaultdict(lambda: 0)
    for f in features:
        hashed[myhash(f)] += 1

    return hashed


def make_liblinear(features):
    return ' '.join('%d:%d' % kv for kv in sorted(features.items()))


def main():
    if len(sys.argv) != 6:
        print('usage: hoshino.py [i]s-tree [i]ts-align [o]s-words [o]ts-align [o]features', file=sys.stderr)
        return
    
    with \
        open(sys.argv[1]) as fi_tree, \
        open(sys.argv[2]) as fi_align, \
        open(sys.argv[3], 'w') as fo_words, \
        open(sys.argv[4], 'w') as fo_align, \
        open(sys.argv[5], 'w') as fo_features:

        for i, (line_tree, line_align) in enumerate(zip(fi_tree, fi_align)):
            input_tree = (add_span(binarize(Tree(line_tree))))
            input_words = extract_words(input_tree)
            input_pos = extract_pos(input_tree)
            input_align = read_align(line_align, len(input_words))

            reorder = make_reorder(input_tree, input_align)
            for label, left, mid, right in reorder:
                if label == 0:
                    continue
                features = make_features(input_tree, input_words, input_pos, left, mid, right)
                liblin = make_liblinear(features)
                print('%s %s' % (label, liblin), file=fo_features)
            
            output_tree = deepcopy(input_tree)
            convert_tree(output_tree, reorder)
            output_align = convert_align(input_align, reorder)
            
            print(' '.join(extract_words(output_tree)), file=fo_words)
            print(make_align_str(output_align), file=fo_align)

            if (i+1) % 1000 == 0:
                print(i+1, file=sys.stderr)

if __name__ == '__main__':
    main()

