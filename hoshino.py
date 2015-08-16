#!/usr/bin/python3

import sys
from binascii import crc32
from nltk.tree import Tree


def myhash(text):
    bits = 24
    mask = (1 << bits) - 1
    return crc32(text.encode()) & mask


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

    l_span = tree[0].label()[1:3]
    r_span = tree[1].label()[1:3]
    l_align = align[l_span[0]:l_span[1]]
    r_align = align[r_span[0]:r_span[1]]
    m_score = kendall(l_align + r_align)
    w_score = kendall(r_align + l_align)
    if m_score > w_score: label = 'M'
    elif m_score < w_score: label = 'W'
    else: label = '-'
    print(label, l_span[0], l_span[1], r_span[1], l_align, r_align)
    
    if label == 'W':
        ret += [(l_span[0], l_span[1], r_span[1])]

    return ret


def convert_tree(tree, reorder):
    if not isinstance(tree, Tree):
        return
    
    l = tree.label()[1]
    r = tree.label()[2]
    for a, _, b in reorder:
        if l == a and r == b:
            tree[0], tree[1] = tree[1], tree[0]
    for ch in tree:
        convert_tree(ch, reorder)


def convert_align(align, reorder):
    for a, b, c in reorder:
        align = align[:a] + align[b:c] + align[a:b] + align[c:]
    return align


def extract_words(tree):
    if not isinstance(tree, Tree):
        return [tree]
    ret = []
    for ch in tree:
        ret += extract_words(ch)
    return ret

def make_align(align):
    ret = ''
    for a, b in enumerate(align):
        if b >= 0:
            ret += '%d-%d ' % (b,a)
    return ret.strip()


def main():
    if len(sys.argv) != 3:
        print('usage: hoshino.py s-tree ts-align', file=sys.stderr)
        return
    
    with open(sys.argv[1]) as ftree, open(sys.argv[2]) as falign:
        for ltree, lalign in zip(ftree, falign):
            tree = (add_span(binarize(Tree(ltree))))
            #print(tree)
            num_words = tree.label()[2]
            align = read_align(lalign, num_words)
            #print(align)
            reorder = make_reorder(tree, align)
            #print(reorder)
            convert_tree(tree, reorder)
            #print(tree)
            a2 = convert_align(align, reorder)
            #print(a2)
            print(' '.join(extract_words(tree)))
            print(make_align(a2))
            return

if __name__ == '__main__':
    main()

