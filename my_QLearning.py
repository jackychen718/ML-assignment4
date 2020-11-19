from mdptoolbox.mdp import MDP,_computeDimensions
import math as _math
import time as _time

import numpy as _np
import scipy.sparse as _sp

import mdptoolbox.util as _util

class MY_QLearning(MDP):
    
    def __init__(self, transitions, reward, discount, n_iter=10000,skip_check=False):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            # We don't want to send this to MDP because _computePR should not
            #  be run on it, so check that it defines an MDP
            _util.check(transitions, reward)

        # Store P, S, and A
        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.discount = discount

        # Initialisations
        self.Q = _np.zeros((self.S, self.A))
        self.mean_discrepancy = []

    def run(self):
        discrepancy = []

        self.time = _time.time()

        s = _np.random.randint(0, self.S)

        for n in range(1, self.max_iter + 1):

            if (n % 10) == 0:
                s = _np.random.randint(0, self.S)

            pn = _np.random.random()
            if pn < (1 - (1 / _math.log(n + 2))):
                a = self.Q[s, :].argmax()
            else:
                a = _np.random.randint(0, self.A)

            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            #dQ = (1 / _math.sqrt(n + 2)) * delta
            dQ=(1/_math.log(n + 2))*delta
            self.Q[s, a] = self.Q[s, a] + dQ

            s = s_new

            discrepancy.append(_np.absolute(dQ))


            if len(discrepancy) == 100:
                self.mean_discrepancy.append(_np.mean(discrepancy))
                discrepancy = []

            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self._endRun()


    def _endRun(self):
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time
        
        

class MY_QLearning_Large(MDP):
    
    def __init__(self, transitions, reward, discount,random_count=3,n_iter=10000,skip_check=False):
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        if not skip_check:
            _util.check(transitions, reward)

        self.S, self.A = _computeDimensions(transitions)
        self.P = self._computeTransition(transitions)

        self.R = reward

        self.discount = discount
        self.random_count=random_count

        self.Q = _np.zeros((self.S, self.A))
        self.mean_discrepancy = []


    def run(self):
        discrepancy = []

        self.time = _time.time()

        s = _np.random.randint(0, self.S)

        for n in range(1, self.max_iter + 1):

            if (n % self.random_count) == 0:
                s = _np.random.randint(0, self.S)

            pn = _np.random.random()
            if pn < (1 - (1 / _math.log(n + 2))):
                a = self.Q[s, :].argmax()
            else:
                a = _np.random.randint(0, self.A)

            p_s_new = _np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            #dQ = (1 / _math.sqrt(n + 2)) * delta
            dQ=(1/_math.log(n + 2))*delta
            self.Q[s, a] = self.Q[s, a] + dQ

            s = s_new

            discrepancy.append(_np.absolute(dQ))


            if len(discrepancy) == 100:
                self.mean_discrepancy.append(_np.mean(discrepancy))
                discrepancy = []

            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self._endRun()


    def _endRun(self):
        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time