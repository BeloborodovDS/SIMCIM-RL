import torch
import torch.nn as nn
import torch.optim as optim

class DQN():
    def __init__(self,
                 net, target_net, 
                 update_freq, 
                 gamma,
                 exploration,
                 num_updates = 1,
                 double = False,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None):

        self.net = net
        self.target_net = target_net
        self.update_target()
        self.update_freq = update_freq
        self.step = 0
        self.exploration = exploration
        self.num_updates = num_updates
        
        self.gamma = gamma
        self.double = double

        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.RMSprop(net.parameters(), lr, eps=eps, alpha=alpha)
        
    def get_eps(self):
        return self.exploration(self.step)
        
    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def update(self, rollouts):
        self.step += 1
        if self.step % self.update_freq == 0:
            self.update_target()
        
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_replay, _ = rollouts.rewards.size()
        
        _, ready = rollouts.ready()
        if not ready:
            return None
        
        for k in range(self.num_updates):
            obs, recurrent_hidden_states, rewards, actions, masks = rollouts.sample_batch()
            T1, N = obs.shape[:2]

            obs = obs.view(T1*N, *obs_shape)
            hx = recurrent_hidden_states[0].view(N, self.net.recurrent_hidden_state_size)
            # create dummy action for last state
            actions = torch.cat([actions, torch.zeros(1,N,action_shape).to(actions)], dim=0)
            actions = actions.view(T1*N, action_shape)
            m = masks.view(T1*N,1)   
            # rewards: T x N
            rewards = rewards.squeeze(2)
            # masks: T+1 x N
            masks = masks.squeeze(2)

            q_for_actions, _, argmax_q, _ = self.net.evaluate_actions(obs, hx, m, actions)
            q_targ_for_argmax, max_q_targ, _, _ = self.target_net.evaluate_actions(obs, hx, m, argmax_q)

            q_for_actions = q_for_actions.view(T1, N)
            argmax_q = argmax_q.view(T1, N)
            q_targ_for_argmax = q_targ_for_argmax.view(T1, N)
            max_q_targ = max_q_targ.view(T1, N)

            if not self.double:
                target_q = max_q_targ
            else:
                target_q = q_targ_for_argmax

            y = rewards + self.gamma*target_q[1:].detach()*masks[1:]
            q = q_for_actions[:-1]
            loss = (q-y).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.net.parameters(),
                                     self.max_grad_norm)

            self.optimizer.step()

        return loss.item()
