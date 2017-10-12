# -*- coding: utf-8 -*-

class Model:
    """
    the base class for inference model
    """
    LIKELIHOOD_DIFF = 1

    def __init__(self):
        pass

    def infer(self, dataset, soft=False):
        pass

    @classmethod
    def op(cls):
        a = cls.LIKELIHOOD_DIFF