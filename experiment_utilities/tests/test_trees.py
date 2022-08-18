from experiment_utilities.trees import tree_flatten, tree_fill


def test_tree_flatten():
    tree1 = [
        {"foo": (4,
                 [2, 3]),
         "bar": [5, {1: 34,
                     's': 6}]
         },
        [2, 7, (8, 4)]
    ]
    tree2 = [[[3, 5, 6], [1, 7]], [4, 5], 8]
    leaves, structure = tree_flatten(tree2)
    filled_tree = tree_fill(leaves, structure)
    assert filled_tree == tree2