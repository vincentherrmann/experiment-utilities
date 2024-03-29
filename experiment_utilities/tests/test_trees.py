import torch

from experiment_utilities.trees import tree_flatten, tree_fill, tree_map
from experiment_utilities.memory import TreeMemory
from experiment_utilities.trees import tree_map

tree1 = [[[3, 5, 6],
          [1, 7]],
         [4, 5],
         8]

tree2 = [
    {"foo": (4,
             [2, 3]),
     "bar": [5, {1: 34,
                 's': 6}]
     },
    [2, 7, (8, 4)]
]


def test_tree_flatten():
    leaves, structure = tree_flatten(tree1)
    filled_tree = tree_fill(leaves, structure)

    assert filled_tree == tree1

    leaves, structure = tree_flatten(tree2)
    filled_tree = tree_fill(leaves, structure)
    assert filled_tree == tree2


def test_tree_map():
    processed_tree = tree_map(f=lambda x: x+1, tree=tree2)
    assert processed_tree[0]["foo"][0] == 5
    assert processed_tree[1][1] == 8

    added_tree = tree_map(f=lambda x, y, z: x + y - z, tree=tree2, rest=(processed_tree, tree2))
    assert added_tree[0]["foo"][0] == 5
    assert added_tree[1][1] == 8
    pass

def test_tree_memory():
    shape_tree = {"parameters": ((10, 5),
                                 (7, 3)),
                  "reward": (1,),
                  "observations": (4,),
                  "test": [(2, 3), (7, 1, 5)]}
    memory = TreeMemory(size=20, shape_tree=shape_tree)

    for i in range(30):
        data_tree = tree_map(lambda shape: torch.rand(list(shape)),
                                    tree=shape_tree, is_leaf=memory.is_leaf)
        memory.store(data_tree)

    data_tree = tree_map(lambda shape: torch.rand([15] + list(shape)),
                         tree=shape_tree, is_leaf=memory.is_leaf)
    memory.store_multiple(data_tree)

    tree_batch = memory.sample_batch(8)
    tree_seq = memory.get_sequence(17, 10)
    pass