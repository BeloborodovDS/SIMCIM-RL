import copy
import os
import time
import argparse
import gc
from collections import defaultdict
import json

import numpy as np
import torch

from matplotlib import pyplot as plt

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy, FILMBase
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule

from simcim import SIMGeneratorRandom, SIMCIM, SIMCollection
from utils import read_gbench, read_gset

from args import my_get_args

    
def main():
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    gbench = read_gbench('./data/gbench.txt')

    args = my_get_args()
    print(args)

    config = dict(
        sigma = args.sim_sigma,
        momentum = args.sim_momentum,
        pump_bins = args.sim_bins,
        lag = 1000//args.num_steps,
        rshift = args.sim_rshift, 
        pump_scale = args.sim_scale,
        reward_kind = args.sim_reward,
        continuous = args.sim_continuous,
        span = args.sim_span,
        percentile = args.sim_percentile,
        last_runs = args.sim_perc_len,
        add_linear = not args.sim_no_linear,
        start_pump = args.sim_start,
        static_features = not args.sim_no_static,
        extra_features = not args.sim_no_extra,
        curiosity_num = args.curiosity
    )

    base_kwargs = {'hidden_size': args.hidden_size, 
                   'film_size': 800 * (not args.sim_no_static)}
    if args.relu:
        base_kwargs['activation'] = 'relu'
    base = FILMBase #FILMBase

    if args.gset > 0:
        test_graphs = [args.gset]
    else:
        test_graphs = [1,2,3,4,5]

    #---------------------------------------------------------

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print('Num updates: ', num_updates)

    if args.dry_run:
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    logdata = defaultdict(list)

    if args.gset > 0:
        envs = []
        for g in test_graphs:
            g_ = read_gset('../data/G{}.txt'.format(g), negate=True)
            s = SIMCIM(g_, device=device, batch_size=args.num_processes, **config)
            s.runpump()
            envs.append(s)
        envs = SIMCollection(envs, [gbench[g] for g in test_graphs])
        logdata['bls_bench'] = [gbench[g] for g in test_graphs]
    else:
        envs = SIMGeneratorRandom(800, 0.06, args.num_processes, config, 
                                  keep=args.sim_keep, n_sims=args.sim_nsim, device=device)

    if args.snapshot is None:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space,
            base=base, base_kwargs=base_kwargs)
    else:
        actor_critic, _ = torch.load(os.path.join(args.save_dir, args.algo, args.snapshot + ".pt"))

    actor_critic.to(device)
    print(actor_critic)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                            args.entropy_coef, lr=args.lr,
                            eps=args.eps, alpha=args.alpha,
                            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                        args.value_loss_coef, args.entropy_coef, lr=args.lr,
                        eps=args.eps, max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                            args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()

    print(rollouts.obs.shape, obs.shape)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
        
    eval_envs = []
    for g in test_graphs:
        g_ = read_gset('../data/G{}.txt'.format(g), negate=True)
        s = SIMCIM(g_, device=device, batch_size=args.num_val_processes, **config)
        s.runpump()
        eval_envs.append(s)
    eval_envs = SIMCollection(eval_envs, [gbench[g] for g in test_graphs])
    ref_cuts = [s.lastcuts for s in eval_envs.envs]
    logdata['ref_cuts'] = [e.tolist() for e in ref_cuts]

    stoch_cuts = None

    start = time.time()
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            if args.algo == "acktr":
                # use optimizer's learning rate since it's hard-coded in kfac.py
                update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
            else:
                update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        if args.algo == 'ppo' and args.use_linear_clip_decay:
            agent.clip_param = args.clip_param  * (1 - j / float(num_updates))
            
        # ROLLOUT DATA
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            if 'episode' in infos[0].keys():
                rw = np.mean([e['episode']['r'] for e in infos])
                logdata['episode_rewards'].append(rw.item())
                if args.gset > 0:
                    cuts = [e.lastcuts for e in envs.envs]
                    logdata['train_median'].append([np.median(e).item() for e in cuts])
                    logdata['train_max'].append([np.max(e).item() for e in cuts])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                    for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        #UPDATE AGENT
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, _ = agent.update(rollouts)
        logdata['alosses'].append(action_loss)
        logdata['vlosses'].append(value_loss)

        logdata['train_percentiles'].append(envs.perc.tolist())

        rollouts.after_update()
        
        #CHECKPOINTS
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                        getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + '-' + str(j) + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        
        #LOGGING
        if j % args.log_interval == 0 and len(logdata['episode_rewards']) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: \
                mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}\n".
                format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(logdata['episode_rewards']),
                    np.mean(logdata['episode_rewards'][-10:]),
                    np.median(logdata['episode_rewards'][-10:]),
                    np.min(logdata['episode_rewards'][-10:]),
                    np.max(logdata['episode_rewards'][-10:])))

        #EVALUATION
        if (args.eval_interval is not None and j % args.eval_interval == 0):
            logdata['spumps'] = []

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_val_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_val_processes, 1, device=device)

            eval_done=False

            while not eval_done:
                p = eval_envs.envs[0].old_p
                logdata['spumps'].append(p[:10].cpu().numpy().tolist())

                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=False)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_done = np.all(done)

                eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                        for done_ in done],
                                        dtype=torch.float32,
                                        device=device)

            stoch_cuts = [e.lastcuts for e in eval_envs.envs]
            logdata['stoch_cuts'] = [e.tolist() for e in stoch_cuts]
            logdata['eval_median'].append([np.median(e).item() for e in stoch_cuts])
            logdata['eval_max'].append([np.max(e).item() for e in stoch_cuts])
            
            logdata['test_percentiles'].append(eval_envs.perc.tolist())

            rw = np.mean([e['episode']['r'] for e in infos])
            logdata['eval_episode_rewards'].append(rw.item())

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(logdata['eval_episode_rewards']),
                    np.mean(logdata['eval_episode_rewards'])))

        if j % args.log_interval == 0:
            fn = os.path.join(save_path, args.env_name + ".res")
            with open(fn, 'w') as f:
                json.dump(logdata, f, sort_keys=True, indent=2)

        #VISUALIZATION
        if j % args.vis_interval == 0:
        #if False:
            plt.figure(figsize=(15,10))
            
            plt.subplot(231)
            plt.title('Rewards')
            plt.xlabel('SIM runs')
            plt.plot(logdata['episode_rewards'], c='r', label='mean train')
            plt.plot(np.linspace(0, len(logdata['episode_rewards']), len(logdata['eval_episode_rewards'])), 
                     logdata['eval_episode_rewards'], 'b', label='mean eval')
            plt.legend()
            
            plt.subplot(232)
            plt.plot(logdata['alosses'])
            plt.title('Policy loss')
            
            plt.subplot(233)
            plt.plot(logdata['vlosses'])
            plt.title('Value loss')
            
            plt.subplot(234)
            plt.title('Pumps')
            plt.xlabel('SIM iterations / 10')
            plt.plot(np.array(logdata['spumps']))
            plt.ylim(-0.05,1.1)
            
            plt.subplot(235)
            plt.plot(logdata['train_percentiles'])
            plt.title('Train average percentile')
            
            plt.subplot(236)
            plt.title('Test percentiles')
            plt.plot(logdata['test_percentiles'])
            plt.legend([str(e) for e in test_graphs])
            
            plt.tight_layout()
            plt.savefig('./plots/agent_'+args.env_name+'.pdf')
            plt.clf()
            plt.close()
            gc.collect()
            #plt.show()

            if stoch_cuts is not None:
                fig, axs = plt.subplots(len(ref_cuts), 1, sharex=False, tight_layout=True)
                if len(ref_cuts)==1:
                    axs = [axs]
                for gi in range(len(ref_cuts)):
                    mn = min(ref_cuts[gi])
                    axs[gi].hist(ref_cuts[gi], bins=100, alpha=0.7)
                    dc = stoch_cuts[gi][stoch_cuts[gi] >= mn]
                    if dc.size>0:
                        axs[gi].hist(dc, bins=100, alpha=0.7)
                plt.savefig('./plots/cuts_'+args.env_name+'.pdf')
                plt.clf()
                plt.close()
                gc.collect()
                #plt.show()


if __name__=='__main__':
    main()