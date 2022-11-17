import numpy as np
import torch as torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import scipy

# Replay memory to store transitions, in a cyclical buffer so olded transitions are removed first (Could be interesting to try a prioritized buffer instead)
class ReplayMemory():
    def __init__(self, size, env):
        self.size = size
        self.counter = 0

        obs_shape = env.observation_space.shape

        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.action_probs = np.zeros((size, env.action_space.n), dtype=np.float32)
        self.returns = np.zeros((size, 1), dtype=np.float32)
    
    def store_transition(self, obs, action_prob, r):
        indx = self.counter % self.size

        self.obs[indx] = np.array(obs).copy()
        self.action_probs[indx] = np.array(action_prob).copy()
        self.returns[indx] = np.array(r).copy()

        self.counter += 1
    
    def sample_batch(self, batch_size):
        valid_indxs = np.min([self.counter, self.size])
        batch = np.random.randint(0, valid_indxs, size=batch_size)

        obs = torch.as_tensor(self.obs[batch])
        action_probs = torch.as_tensor(self.action_probs[batch])
        returns = torch.as_tensor(self.returns[batch])

        return obs, action_probs, returns


def train_on_batch(network, optimizer, batch, step, writer):
    # Run network in training mode
    network.train()

    obs = batch[0]
    action_probs = batch[1]
    returns = batch[2]

    _, pred_vals = network(torch.unsqueeze(obs, 1))
    policy_logits = network.logits(torch.unsqueeze(obs, 1))

    value_loss = F.mse_loss(pred_vals, returns)
    policy_loss = F.cross_entropy(policy_logits, action_probs)

    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        grads = [torch.flatten(params.grad.detach()) for params in network.parameters()]
        grads = torch.cat(grads).view(-1,1)
        writer.add_scalar("losses/grads", np.linalg.norm(grads.numpy()), step)
        writer.add_histogram("preds/batch_pred_value", pred_vals, step)
        writer.add_histogram("preds/batch_actual_value", returns, step)

    return value_loss, policy_loss

# Compute discounts
# https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
def compute_returns(rewards, discount):
    """
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = scipy.signal.lfilter(b, a, x=r)
    return y[::-1]