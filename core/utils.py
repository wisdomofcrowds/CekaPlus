# -*- coding: utf-8 -*-
import numpy

def gen_rand_sum_one(size):
    """
    generate size random numbers that sum up to one
    :param size: number of random numbers
    :return: list of numbers
    """
    lst = []
    sum = 0
    for i in range(0, size):
        r = numpy.random.random()
        sum += r
        lst.append(r)
    lst2 = [x/sum for x in lst]
    return lst2


