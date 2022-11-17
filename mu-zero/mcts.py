import numpy as np
from copy import deepcopy
import torch
import scipy

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Node:
    """
    Transition node. Represents the child of a node as a result of an action being taken. 
    """
    def __init__(self, h, action=None, prior=0):
        self.h = h
        self.action = action
        self.reward = 0
        self.done = False

        self.parent = None
        self.children = {}

        self.value_sum = 0.0
        self.visits = 0
        self.prior = prior
    
    def __repr__(self):
        return f"Node: (s:{self.state}, val:{self.value():0.2f}, visits: {self.visits}, p:{self.prior:0.2f})"
    
    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def update_stats(self, val):
        self.visits += 1
        self.value_sum += val

class MCTS_MuZero:
    """
    MCTS using a similar approach to AlphaZero: https://arxiv.org/pdf/1712.01815.pdf
    """
    def __init__(self, cur_state, env, network, hparams):
        self.root = Node(cur_state, env)
        self.network = network
        self.action_size = env.action_space.n
        self.iters: int = hparams["search_iters"]
        self.discount = hparams.get("discount", 0.99) 
        self.pc_base = 19652
        self.pc_init = 2.5
    
    def search(self):
        ## Expand root + add noise for exploration
        self._expand(self.root)
        dirichlet_noise(0.25, 0.25, self.root)
        for _ in range(self.iters):
            # Tree Policy
            # Run through the tree and recursively select the best nodes with respect to their `PUCT` values
            next_node = self.root
            while next_node.is_expanded():
                next_node = self._select(next_node)

            # Expand the leaf node by evaluating network for policy probs and value at state, sample most likely action
            value = self._expand(next_node)

            # Backup the value of this node or the reward if its a terminal state (TODO: Seems suspicious for non terminal rewards)
            self._backup(next_node, value)
        
        # print(f"Actions visits: {[child.visits for child in root.children.values()]}")
        policy = self._pi()
        return self._best_action(policy), policy
    
    def forward(self, action):
        self.root = self.root.children[action]


    # "Most Robust Child" sampling: http://www.incompleteideas.net/609%20dropbox/other%20readings%20and%20resources/MCTS-survey.pdf
    def _best_action(self, policy_probs):
        return np.random.choice(range(len(policy_probs)),p=policy_probs)
    
    def _pi(self):
        visit_counts = [(action, child.visits) for action, child in self.root.children.items()]
        visit_counts = [x[1] for x in sorted(visit_counts)]
        av = np.array(visit_counts).astype(np.float64)
        return softmax(av)

    def _expand(self, node: Node) -> Node:
        # Expand and add children with predicted prior
        prior, value = self.network.predict(torch.Tensor([node.state]))
        prior, value = prior.detach().numpy(), value.detach().numpy()[0]

        # Get current env to sample actions from
        for action in range(self.action_size):
            h_n, r = self.network.dynamics(node.h, action)

            # Update tree with transition
            next_node = Node(h_n, action=action, prior=prior[action])
            next_node.parent = node
            next_node.reward = r
            node.children[action] = next_node

        return value


    # Detailed here: https://web.stanford.edu/~surag/posts/alphazero.html
    def _select(self, node: Node) -> Node:

        # Get children and compute state-action values
        children: list[Node] = list(node.children.values())
        puct_vals = np.array([self._PUCT(node, child) for child in children])

        # Get argmax, np-way (no tie breaking)
        return children[np.argmax(puct_vals)]

    def _PUCT(self, parent: Node, child: Node) -> float:
        # Action value
        child_value = child.reward + self.discount * child.value()

        # Exploration bonus
        # explore = self.pc_base + np.log((parent.visits + self.pc_base) / self.pc_base) # Exploration should steadily decrease
        explore = child.prior
        explore *= np.sqrt(parent.visits + 1) / (child.visits + 1)

        return child_value + explore


    def _backup(self, node: Node, value: float) -> None:
        v = value
        while node:
            v = node.reward + self.discount * value
            node.update_stats(v)
            node = node.parent

def dirichlet_noise(alpha, exploration_frac, node: Node):
  actions = node.children.keys()
  noise = np.random.gamma(alpha, 1, len(actions))
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - exploration_frac) + n * exploration_frac
