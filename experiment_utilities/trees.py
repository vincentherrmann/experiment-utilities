from functools import reduce
# These functions are heavily inspired by jax pytrees


def tree_flatten(tree, is_leaf=None, reference_tree=None):
    def traverse_leaves(node, is_leaf=None, reference_node=None):
        leaves = []
        if is_leaf is not None and is_leaf(node):
            return [node]
        elif type(node) is list or type(node) is tuple:
            for i in range(len(node)):
                leaves.extend(traverse_leaves(node[i], is_leaf,
                                              reference_node[i] if reference_node is not None else None))
        elif type(node) is dict:
            keys = node.keys() if reference_node is None else reference_node.keys()
            for key in keys:
                child = node[key]
                leaves.extend(traverse_leaves(child, is_leaf,
                                              reference_node[key] if reference_node is not None else None))
        else:
            return [node]
        return leaves

    def traverse_structure(node, is_leaf=None, reference_node=None):
        if is_leaf is not None and is_leaf(node):
            return None
        elif type(node) is list or type(node) is tuple:
            structure = [traverse_structure(node[i], is_leaf, reference_node[i] if reference_node is not None else None)
                         for i in range(len(node))]
            if type(node) is tuple:
                structure = tuple(structure)
        elif type(node) is dict:
            keys = node.keys() if reference_node is None else reference_node.keys()
            structure = {key: traverse_structure(node[key], is_leaf,
                                                 reference_node[key] if reference_node is not None else None)
                         for key in keys}
        else:
            return None
        return structure

    leaves = traverse_leaves(tree, is_leaf, reference_tree)
    structure = traverse_structure(tree, is_leaf, reference_tree)
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
    all_leaves = [leaves] + [tree_flatten(r, is_leaf, tree)[0] for r in rest]
    if len(rest) > 0:
        assert len(leaves) == len(all_leaves[1])
    processed_leaves = [f(*xs) for xs in zip(*all_leaves)]
    processed_tree = tree_fill(processed_leaves, structure, is_leaf)
    return processed_tree

def tree_modify(f, tree, rest=[], is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [tree_flatten(r, is_leaf, tree)[0] for r in rest]
    for xs in zip(*all_leaves):
        f(*xs)


def tree_reduce(f, tree, is_leaf=None):
    leaves, structure = tree_flatten(tree, is_leaf)
    r = reduce(f, leaves)
    return r

