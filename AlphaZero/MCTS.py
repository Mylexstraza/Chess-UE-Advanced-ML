import logging
import math
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.s = None
        self.game = game
        self.nnet = nnet
        self.args = args
        self.cpt = 0
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    
    def getActionProb(self, canonicalBoard, has_moved, history, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.cpt = 0
            self.compteur = {}
            self.search(canonicalBoard, has_moved, history)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        """
        valids = self.game.getValidMoves(canonicalBoard, 1, has_moved)
        a = [index for index, valeur in enumerate(counts) if valeur != 0]
        b = [index for index, valeur in enumerate(valids) if valeur != 0]
        difference = [element for element in a if element not in b]

        self.game.verify_valid_moves(canonicalBoard, a, b, difference)
        """

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))            
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, has_moved, hist):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
       
        history = hist.copy()

        if self.cpt >= self.args.maxiters:
            #print("max steps reached in mcts")
            return "tie"
        
        s = self.game.stringRepresentation(canonicalBoard)
        history.append(canonicalBoard)
        if len(history) > 9:
                history.pop(0)

        if s in self.compteur:
            self.compteur[s] += 1
            if self.compteur[s] == 3:
                return "tie"
        else:
            self.compteur[s] = 1

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1, has_moved)

        if self.Es[s] == "tie":
            return "tie"
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]
        
        if s not in self.Ps:
            # leaf node
            boards_for_nn = np.concatenate([self.game.board_to_nn_input(ss) for ss in history], axis=0)
            self.Ps[s], v = self.nnet.predict(boards_for_nn)
            valids = self.game.getValidMoves(canonicalBoard, 1, has_moved)
            
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
        

        a = best_act

        next_s, has_moved, next_player = self.game.getNextState(canonicalBoard, 1, a, has_moved)

        if self.game.compare_echiquiers(canonicalBoard, next_s):
            self.cpt = 0
        else:
            self.cpt += 1

        next_s, has_moved = self.game.getCanonicalForm(next_s, has_moved, next_player)


        #print(self.game.getGameEnded(canonicalBoard, 1, next_has_moved))
        v = self.search(next_s, has_moved, history)
        if v == "tie":
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = 0
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = 0
                self.Nsa[(s, a)] = 1
            self.Ns[s] += 1

            return v
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        
        return -v

