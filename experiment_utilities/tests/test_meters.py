from unittest import TestCase
import pytest
from experiment_utilities.meters import AverageMeter
import torch

class TestAverageMeter(TestCase):
    def test_running_variance(self):
        m = AverageMeter()
        l = torch.rand(100)

        for value in l:
            m.update(value.item())

        assert pytest.approx(m.var, l.var().item())

