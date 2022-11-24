from functools import reduce
# These functions are heavily inspired by jax pytrees


def tree_flatten(tree, is_leaf=None):
    def traverse_leaves(node, is_leaf=None):
        leaves = []
        if is_leaf is not None and is_leaf(node):
            return [node]
        elif type(node) is list or type(node) is tuple:
            for child in node:
                leaves.extend(traverse_leaves(child, is_leaf))
        elif type(node) is dict:
            for key, child in node.items():
                leaves.extend(traverse_leaves(child, is_leaf))
        else:
            return [node]
        return leaves

    def traverse_structure(node, is_leaf=None):
        if is_leaf is not None and is_leaf(node):
            return None
        elif type(node) is list:
            structure = [traverse_structure(child, is_leaf) for child in node]
        elif type(node) is tuple:
            structure = tuple([traverse_structure(child, is_leaf) for child in node])
        elif type(node) is dict:
            structure = {key: traverse_structure(child, is_leaf) for key, child in node.items()}
        else:
            return None
        return structure

    leaves = traverse_leaves(tree, is_leaf)
    structure = traverse_structure(tree, is_leaf)
    return leaves, structure


def tree_fill(leaves, structure, is_leaf=None):
    if is_leaf is not None and is_leaf(structure):
        return leaves.pop(0)
    if type(structure) is list:
        filled_structure = [tree_fill(leaves, child, is_leaf) for child in structure]
    elif type(structure) is tuple:
        filled_structure = tuple([tree_fill(leaves, child, is_leaf) for child in structure])
    elif type(structure) is dict:
        filled_structure = {key: tree_fill(leaves, child, is_leaf) for key, child in structure.items()}
    else:
        return leaves.pop(0)
    return filled_structure


def tree_map(f, tree, rest=[], is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [tree_flatten(r, is_leaf)[0] for r in rest]
    if len(rest) > 0:
        assert len(leaves) == len(all_leaves[1])
    processed_leaves = [f(*xs) for xs in zip(*all_leaves)]
    processed_tree = tree_fill(processed_leaves, structure, is_leaf)
    return processed_tree

def tree_modify(f, tree, rest=[], is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [tree_flatten(r, is_leaf)[0] for r in rest]
    for xs in zip(*all_leaves):
        f(*xs)


def tree_reduce(f, tree, is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    r = reduce(f, leaves)
    return r

