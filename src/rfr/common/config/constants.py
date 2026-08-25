"""Shared Stage-1 approximation costs used by final evaluation."""

GELU_COST = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
SOFTMAX_COST = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
