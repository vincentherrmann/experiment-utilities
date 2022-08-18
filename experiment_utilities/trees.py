TREE_CONTAINER_TYPES = [list, tuple, dict]


def tree_flatten(tree, is_leaf=None):
    def traverse_leaves(node):
        leaves = []
        if type(node) in TREE_CONTAINER_TYPES:
            for child in node:
                l = traverse_leaves(child)
                leaves.extend(l)
        else:
            return [node]
        return leaves

    def traverse_structure(node):
        if type(node) in TREE_CONTAINER_TYPES:
            nested_list = [traverse_structure(child) for child in node]
        else:
            return None
        return nested_list

    # def traverse(node, structure=None):

        # if type(node) in TREE_CONTAINER_TYPES:
        #     for child in node:
        #         traverse(child)
        # else:
        #     print(node)
        #     return None

    leaves = traverse_leaves(tree)
    structure = traverse_structure(tree)
    return leaves, structure


def tree_fill(leaves, structure):
    if type(structure) in TREE_CONTAINER_TYPES:
        filled_structure = [tree_fill(leaves, child) for child in structure]
    else:
        return leaves.pop(0)
    return filled_structure

