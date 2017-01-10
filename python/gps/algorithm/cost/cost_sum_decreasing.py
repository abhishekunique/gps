""" This file defines a cost sum of arbitrary other costs. """
import copy

from gps.algorithm.cost.config import COST_SUM
from gps.algorithm.cost.cost import Cost


class CostSumDecrease(Cost):
    """ A wrapper cost function that adds other cost functions. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']

        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))

    def eval(self, sample, itr):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)

        # Compute weighted sum of each cost value and derivatives.
        # weights = copy.copy(self._weights)
        # if itr<5:
        #     weights[0] = 0.3
        #     weights[1] = 1.0
        # else:
        #     weights[0] = 1.0
        #     weights[1] = 1.0/((itr + 1)**(0.3))
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            weight = self._weights[i] * (0.5 ** i)
            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight
        return l, lx, lu, lxx, luu, lux
