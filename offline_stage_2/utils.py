import numpy as np
import pickle, torch
from torch.distributions import Categorical
import logging
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import envs.multiagent_particle_envs.multiagent.scenarios as scenarios
from envs.multiagent_particle_envs.multiagent.environment import MultiAgentEnv
from envs.multiagent_particle_envs.opponent_policy import get_all_oppo_policies as get_all_oppo_policies_mpe
from open_spiel.python import rl_environment
from envs.markov_soccer.opponent_policy import get_all_oppo_policies as get_all_oppo_policies_soccer
from envs.markov_soccer.soccer_state import get_two_state


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()


def get_env_and_oppo(config, oppo_policy):
    env_and_oppo = dict()
    if config["ENV_TYPE"] == "PA":
        scenario = scenarios.load(config["SCENARIO"]).Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None, shared_viewer=False)
        seed = config["SEED_RES"]
        env.seed(seed)
        all_oppo_policies = get_all_oppo_policies_mpe(env, oppo_policy, config["OPPO_INDEX"][0], config["NUM_STEPS"], config["EVAL_DEVICE"])
        env_and_oppo["env"] = env
        env_and_oppo["oppo_policy"] = all_oppo_policies
        return env_and_oppo
    elif config['ENV_TYPE'] == 'MS':
        seed = config["SEED_RES"]
        env = rl_environment.Environment(config["SCENARIO"])
        env.seed(seed)
        all_oppo_policies = get_all_oppo_policies_soccer(oppo_policy, config["OPPO_INDEX"][0], config["EVAL_DEVICE"])
        env_and_oppo["env"] = env
        env_and_oppo["oppo_policy"] = all_oppo_policies
        return env_and_oppo


def eval_episode_rtg(
        env,
        env_type,
        agent_obs_dim,
        oppo_obs_dim,
        act_dim,
        c_dim,
        encoder,
        decoder,
        oppo_policy,
        agent_index,
        oppo_index,
        num_steps=100,
        reward_scale=1000.,
        target_rtg=None,
        eval_mode='normal',
        agent_obs_mean=0.,
        agent_obs_std=1.,
        oppo_obs_mean=0.,
        oppo_obs_std=1.,
        oppo_context_window=None,
        device="cuda",
        obs_normalize=True,
):
    encoder.eval()
    encoder.to(device=device)
    K = encoder.K
    decoder.eval()
    decoder.to(device=device)

    agent_obs_mean = torch.from_numpy(agent_obs_mean).to(device=device)
    agent_obs_std = torch.from_numpy(agent_obs_std).to(device=device)
    oppo_obs_mean = torch.from_numpy(oppo_obs_mean).to(device=device)
    oppo_obs_std = torch.from_numpy(oppo_obs_std).to(device=device)

    if env_type == 'MS':
        time_step = env.reset()
        _, _, rel_state1, rel_state2 = get_two_state(time_step)
        obs_n = [rel_state1, rel_state2]
    else:
        obs_n = env.reset()
    if eval_mode == 'noise':
        for i in agent_index:
            obs_n[i] = obs_n[i] + np.random.normal(0, 0.1, size=obs_n[i].shape)

    if oppo_context_window != None:
        oppo_embeds, oppo_mask = [], []
        for oppo_trajs in oppo_context_window:
            n_o_oppo, a_oppo, r_oppo, _, _, timestep_oppo = oppo_trajs
            es = np.random.randint(0, n_o_oppo.shape[0])  # 返回一个随机整数
            oppo_embeds_, oppo_mask_ = encoder.get_tokens(
                n_o_oppo[es:es + K].to(device=device, dtype=torch.float32),
                a_oppo[es:es + K].to(device=device, dtype=torch.float32),
                r_oppo[es:es + K].to(device=device, dtype=torch.float32),
                timestep_oppo[es:es + K].to(device=device, dtype=torch.long),
                attention_mask=None,
            )
            oppo_embeds.append(oppo_embeds_)
            oppo_mask.append(oppo_mask_)
        oppo_embeds = torch.cat(oppo_embeds, dim=1).contiguous()
        oppo_mask = torch.cat(oppo_mask, dim=1).contiguous()
    else:
        oppo_embeds, oppo_mask = None, None

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_list_n = [None for _ in oppo_index + agent_index]
    act_list_n = [None for _ in oppo_index + agent_index]
    r_list_n = [None for _ in oppo_index + agent_index]
    target_rtg_list_n = [None for _ in oppo_index + agent_index]
    timestep_list_n = [None for _ in oppo_index + agent_index]
    for i in oppo_index + agent_index:
        obs_list_n[i] = torch.from_numpy(obs_n[i]).reshape(1, agent_obs_dim if i in agent_index else oppo_obs_dim).to(
            device=device, dtype=torch.float32)
        act_list_n[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        r_list_n[i] = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = target_rtg
        target_rtg_list_n[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timestep_list_n[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return = [0. for _ in oppo_index + agent_index]
    true_steps = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_index + agent_index]
        for i in agent_index:
            # add padding
            act_list_n[i] = torch.cat([act_list_n[i], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[i] = torch.cat([r_list_n[i], torch.zeros(1, device=device)])
            if obs_normalize:
                action = decoder.get_action(
                    (obs_list_n[i].to(dtype=torch.float32) - agent_obs_mean) / agent_obs_std,
                    act_list_n[i].to(dtype=torch.float32),
                    r_list_n[i].to(dtype=torch.float32),
                    target_rtg_list_n[i].to(dtype=torch.float32),
                    timestep_list_n[i].to(dtype=torch.long),
                    oppo_embeds,
                    oppo_mask,
                )
            else:
                action = decoder.get_action(
                    obs_list_n[i].to(dtype=torch.float32),
                    act_list_n[i].to(dtype=torch.float32),
                    r_list_n[i].to(dtype=torch.float32),
                    target_rtg_list_n[i].to(dtype=torch.float32),
                    timestep_list_n[i].to(dtype=torch.long),
                    oppo_embeds,
                    oppo_mask,
                )
            if env_type == "PA":
                action = torch.nn.Softmax(dim=0)(action)
                action_index = torch.argmax(action)
                action = torch.eye(act_dim, dtype=torch.float32)[action_index]
                act = action.detach().clone().cpu().numpy()
                act = np.concatenate([act, np.zeros(c_dim)])
            elif env_type == 'MS':
                action_prob = torch.nn.Softmax(dim=0)(action)
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = torch.eye(act_dim, dtype=torch.float32)[act]
            act_n[i] = act
            act_list_n[i][-1] = action

        for j in oppo_index:
            act_list_n[j] = torch.cat([act_list_n[j], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[j] = torch.cat([r_list_n[j], torch.zeros(1, device=device)])
            if env_type == "PA":
                action = act = oppo_policy.action(obs_n[j])
            elif env_type == 'MS':
                action_prob = oppo_policy(torch.tensor(obs_n[j], dtype=torch.float32, device=device))
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = np.eye(act_dim, dtype=np.float32)[act]
            act_n[j] = act
            cur_action = torch.from_numpy(action[:act_dim]).to(device=device, dtype=torch.float32).reshape(1, act_dim)
            act_list_n[j][-1] = cur_action

        if env_type == 'MS':
            act_n = np.array(act_n)
            time_step = env.step(act_n)
            rew1, rew2 = time_step.rewards[0], time_step.rewards[1]
            reward_n = [rew1, rew2]
            _, _, rel_state1_, rel_state2_ = get_two_state(time_step)
            obs_n = [rel_state1_, rel_state2_]
            done_ = (time_step.last() == True)
            done_n = [done_, done_]
            info_n = {}
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)

        for i in oppo_index + agent_index:
            cur_obs = torch.from_numpy(obs_n[i]).to(device=device, dtype=torch.float32).reshape(1,
                                                                                                agent_obs_dim if i in agent_index else oppo_obs_dim)
            obs_list_n[i] = torch.cat([obs_list_n[i], cur_obs], dim=0)
            r_list_n[i][-1] = reward_n[i]

            if eval_mode != 'delayed':
                pred_return = target_rtg_list_n[i][0, -1] - (reward_n[i] / reward_scale)
            else:
                pred_return = target_rtg_list_n[i][0, -1]
            target_rtg_list_n[i] = torch.cat(
                [target_rtg_list_n[i], pred_return.reshape(1, 1)], dim=1)
            timestep_list_n[i] = torch.cat(
                [timestep_list_n[i],
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            episode_return[i] += reward_n[i]
        true_steps += 1
        if done_n[0] or done_n[1]:
            break

    for j in oppo_index:
        if obs_normalize:
            n_o_oppo = (obs_list_n[j] - oppo_obs_mean) / oppo_obs_std
        else:
            n_o_oppo = obs_list_n[j]
        a_oppo = act_list_n[j]
        r_oppo = r_list_n[j]
        timestep_oppo = timestep_list_n[j]
    steps_ = min(num_steps, true_steps)
    if oppo_context_window == None:
        oppo_context_window = [(n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1])]
    else:
        oppo_context_window.append((
                                   n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                   timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1]))

    average_epi_return = np.mean([episode_return[k] for k in agent_index])

    return average_epi_return, oppo_context_window


def eval_episodes(env, oppo_policy, config, oppo_id, oppo_name):
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    if config['OBS_NORMALIZE']:
        agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
        oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
    else:
        agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
        oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_eval_episodes, num_steps = config["NUM_EVAL_EPISODES"], config["NUM_STEPS"]
    if oppo_name in config["SEEN_OPPO_POLICY"]:
        target_rtg = config["OPPO_TARGET"][oppo_id]
    else:
        target_rtg = np.mean(config["OPPO_TARGET"])
    reward_scale = config["REWARD_SCALE"]
    env_type = config["ENV_TYPE"]
    if isinstance(oppo_name, tuple):
        oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
    else:
        oppo_name_ = oppo_name
    c_dim = config["C_DIM"]
    device = config["EVAL_DEVICE"]
    eval_mode = config['EVAL_MODE']
    ocw_size = config["OCW_SIZE"]
    
    def fn(encoder, decoder):
        returns = []
        oppo_context_window = None
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, oppo_context_window_new = eval_episode_rtg(
                    env,
                    env_type,
                    agent_obs_dim,
                    oppo_obs_dim,
                    act_dim,
                    c_dim,
                    encoder,
                    decoder,
                    oppo_policy,
                    agent_index,
                    oppo_index,
                    num_steps=num_steps,
                    reward_scale=reward_scale,
                    target_rtg=target_rtg / reward_scale,
                    eval_mode=eval_mode,
                    agent_obs_mean=agent_obs_mean,
                    agent_obs_std=agent_obs_std,
                    oppo_obs_mean=oppo_obs_mean,
                    oppo_obs_std=oppo_obs_std,
                    oppo_context_window=oppo_context_window,
                    device=device,
                    obs_normalize=config['OBS_NORMALIZE'],
                    )
                oppo_context_window = oppo_context_window_new
                oppo_context_window = oppo_context_window[-ocw_size:]
            returns.append(ret)
        return {
            f'{oppo_name_}_target_{target_rtg:.3f}_return_mean': np.mean(returns),
        }, np.mean(returns)
    return fn


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    if len(x) == 0:
        return discount_cumsum
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def get_batch(offline_data, config_dict):
    agent_obs_dim = config_dict["AGENT_OBS_DIM"]
    oppo_obs_dim = config_dict["OPPO_OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    num_oppo_policy = config_dict["NUM_OPPO_POLICY"]
    num_trajs_list = config_dict["NUM_TRAJS"]
    batch_size = config_dict["BATCH_SIZE"]
    K_encoder = config_dict["K"]
    K_decoder = config_dict["K"]
    num_steps = config_dict["NUM_STEPS"]
    if config_dict["OBS_NORMALIZE"]:
        agent_obs_mean_list = config_dict["AGENT_OBS_MEAN"]
        agent_obs_std_list = config_dict["AGENT_OBS_STD"]
        oppo_obs_mean_list = config_dict["OPPO_OBS_MEAN"]
        oppo_obs_std_list = config_dict["OPPO_OBS_STD"]
    reward_scale = config_dict["REWARD_SCALE"]
    device = config_dict["DEVICE"]
    ocw_size = config_dict["OCW_SIZE"]
    def fn(batch_size=batch_size, max_len_e=K_encoder, max_len_d=K_decoder):
        n_o_e, a_e, r_e, timesteps_e, mask_e = [], [], [], [], []
        o_d, a_d, r_d, rtg_d, timesteps_d, mask_d = [], [], [], [], [], []
        o_e_d, a_e_d = [], []
        for i in range(num_oppo_policy):
            batch_inds = np.random.choice(
                np.arange(num_trajs_list[i]),
                size=batch_size,
                replace=False,
            )
            for k in range(batch_size):
                traj = offline_data[i][batch_inds[k]]
                ds = np.random.randint(0, traj[0]['rewards'].shape[0])
                o_d.append(traj[0]['observations'][ds:ds + max_len_d].reshape(1, -1, agent_obs_dim))
                a_d.append(traj[0]['actions'][ds:ds + max_len_d].reshape(1, -1, act_dim))
                r_d.append(traj[0]['rewards'][ds:ds + max_len_d].reshape(1, -1, 1))
                timesteps_d.append(np.arange(ds, ds + o_d[-1].shape[1]).reshape(1, -1))
                timesteps_d[-1][timesteps_d[-1] >= num_steps] = num_steps - 1  # padding cutoff
                rtg_d.append(discount_cumsum(traj[0]['rewards'][ds:], gamma=1.)[:o_d[-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg_d[-1].shape[1] <= o_d[-1].shape[1]:
                    rtg_d[-1] = np.concatenate([rtg_d[-1], np.zeros((1, 1, 1))], axis=1)
                
                tlen_d = o_d[-1].shape[1]
                
                o_d[-1] = np.concatenate([np.zeros((1, max_len_d - tlen_d, agent_obs_dim)), o_d[-1]], axis=1)
                if config_dict["OBS_NORMALIZE"]:
                    o_d[-1] = (o_d[-1] - agent_obs_mean_list[i]) / agent_obs_std_list[i]
                a_d[-1] = np.concatenate([np.ones((1, max_len_d - tlen_d, act_dim)) * -10., a_d[-1]], axis=1)
                r_d[-1] = np.concatenate([np.zeros((1, max_len_d - tlen_d, 1)), r_d[-1]], axis=1)
                rtg_d[-1] = np.concatenate([np.zeros((1, max_len_d - tlen_d, 1)), rtg_d[-1]], axis=1) / reward_scale
                timesteps_d[-1] = np.concatenate([np.zeros((1, max_len_d - tlen_d)), timesteps_d[-1]], axis=1)
                mask_d.append(np.concatenate([np.zeros((1, max_len_d - tlen_d)), np.ones((1, tlen_d))], axis=1))
                
                o_e_d.append(traj[1]['observations'][ds:ds + max_len_d].reshape(1, -1, oppo_obs_dim))
                a_e_d.append(traj[1]['actions'][ds:ds + max_len_d].reshape(1, -1, act_dim))
                
                o_e_d[-1] = np.concatenate([np.zeros((1, max_len_d - tlen_d, oppo_obs_dim)), o_e_d[-1]], axis=1)
                if config_dict["OBS_NORMALIZE"]:
                    o_e_d[-1] = (o_e_d[-1] - oppo_obs_mean_list[i]) / oppo_obs_std_list[i]
                a_e_d[-1] = np.concatenate([np.ones((1, max_len_d - tlen_d, act_dim)) * -10., a_e_d[-1]], axis=1)
                
                oppo_batch_inds = np.random.choice(
                    np.arange(num_trajs_list[i]),
                    size=ocw_size,
                    replace=False,
                )
                n_o_e_, a_e_, r_e_, timesteps_e_, mask_e_ = [], [], [], [], []
                for j in range(ocw_size):
                    oppo_traj = offline_data[i][oppo_batch_inds[j]]
                    es = np.random.randint(0, oppo_traj[1]['rewards'].shape[0])
                    n_o_e_.append(oppo_traj[1]['next_observations'][es:es+max_len_e].reshape(1, -1, oppo_obs_dim))
                    a_e_.append(oppo_traj[1]['actions'][es:es+max_len_e].reshape(1, -1, act_dim))
                    r_e_.append(oppo_traj[1]['rewards'][es:es+max_len_e].reshape(1, -1, 1))
                    timesteps_e_.append(np.arange((es+1), (es+1+n_o_e_[-1].shape[1])).reshape(1, -1))
                    timesteps_e_[-1][timesteps_e_[-1] >= (num_steps+1)] = num_steps
                    
                    tlen_e = n_o_e_[-1].shape[1]
                    
                    n_o_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, oppo_obs_dim)), n_o_e_[-1]], axis=1)
                    if config_dict["OBS_NORMALIZE"]:
                        n_o_e_[-1] = (n_o_e_[-1] - oppo_obs_mean_list[i]) / oppo_obs_std_list[i]
                    a_e_[-1] = np.concatenate([np.ones((1, max_len_e - tlen_e, act_dim)) * -10., a_e_[-1]], axis=1)
                    r_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e, 1)), r_e_[-1]], axis=1)
                    timesteps_e_[-1] = np.concatenate([np.zeros((1, max_len_e - tlen_e)), timesteps_e_[-1]], axis=1)
                    mask_e_.append(np.concatenate([np.zeros((1, max_len_e - tlen_e)), np.ones((1, tlen_e))], axis=1))
                n_o_e.append(np.concatenate(n_o_e_, axis=1))
                a_e.append(np.concatenate(a_e_, axis=1))
                r_e.append(np.concatenate(r_e_, axis=1))
                timesteps_e.append(np.concatenate(timesteps_e_, axis=1))
                mask_e.append(np.concatenate(mask_e_, axis=1))

        o_d = torch.from_numpy(np.concatenate(o_d, axis=0)).to(dtype=torch.float32, device=device)
        a_d = torch.from_numpy(np.concatenate(a_d, axis=0)).to(dtype=torch.float32, device=device)
        r_d = torch.from_numpy(np.concatenate(r_d, axis=0)).to(dtype=torch.float32, device=device)
        rtg_d = torch.from_numpy(np.concatenate(rtg_d, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_d = torch.from_numpy(np.concatenate(timesteps_d, axis=0)).to(dtype=torch.long, device=device)
        mask_d = torch.from_numpy(np.concatenate(mask_d, axis=0)).to(device=device)

        n_o_e = torch.from_numpy(np.concatenate(n_o_e, axis=0)).to(dtype=torch.float32, device=device)
        a_e = torch.from_numpy(np.concatenate(a_e, axis=0)).to(dtype=torch.float32, device=device)
        r_e = torch.from_numpy(np.concatenate(r_e, axis=0)).to(dtype=torch.float32, device=device)
        timesteps_e = torch.from_numpy(np.concatenate(timesteps_e, axis=0)).to(dtype=torch.long, device=device)
        mask_e = torch.from_numpy(np.concatenate(mask_e, axis=0)).to(device=device)
        o_e_d = torch.from_numpy(np.concatenate(o_e_d, axis=0)).to(dtype=torch.float32, device=device)
        a_e_d = torch.from_numpy(np.concatenate(a_e_d, axis=0)).to(dtype=torch.float32, device=device)
        
        return n_o_e, a_e, r_e, timesteps_e, mask_e, o_d, a_d, r_d, rtg_d, timesteps_d, mask_d, o_e_d, a_e_d
    
    return fn


def load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, config_dict):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path), 'rb') as f:
        data = pickle.load(f)
        data_o = data["observations"]
        data_a = data["actions"]
        data_r = data["rewards"]
        data_o_next = data["next_observations"]
    num_oppo_policy = len(data_o)
    config_dict["NUM_OPPO_POLICY"] = num_oppo_policy
    num_age_policy = len(data_o[0])      # 受控代理的策略 的个数
    returns_against_oppo_list = [[[] for __ in range(num_age_policy)] for _ in range(num_oppo_policy)]
    data_list = [[] for _ in range(num_oppo_policy)]
    for i in range(num_oppo_policy):
        num_agent_policy = len(data_o[i])
        for j in range(num_agent_policy):
            num_epis = len(data_o[i][j])
            for e in range(num_epis):
                num_steps = len(data_o[i][j][e])
                for agent in agent_index:
                    agent_o_ep = []
                    agent_a_ep = []
                    agent_r_ep = []
                    for oppo in oppo_index:
                        oppo_o_ep = []
                        oppo_a_ep = []
                        oppo_r_ep = []
                        oppo_o_next_ep = []
                        for k in range(num_steps):
                            agent_o_ep.append(np.array(data_o[i][j][e][k][agent]))
                            agent_a_ep.append(np.array(data_a[i][j][e][k][agent])[:act_dim])
                            agent_r_ep.append(np.array(data_r[i][j][e][k][agent]))
                            oppo_o_ep.append(np.array(data_o[i][j][e][k][oppo]))
                            oppo_a_ep.append(np.array(data_a[i][j][e][k][oppo])[:act_dim])
                            oppo_r_ep.append(np.array(data_r[i][j][e][k][oppo]))
                            oppo_o_next_ep.append(np.array(data_o_next[i][j][e][k][oppo]))
                        data_list[i].append([
                            {
                                "observations": np.array(agent_o_ep),
                                "actions": np.array(agent_a_ep),
                                "rewards": np.array(agent_r_ep),
                            },
                            {
                                "observations": np.array(oppo_o_ep),
                                "actions": np.array(oppo_a_ep),
                                "rewards": np.array(oppo_r_ep),
                                "next_observations": np.array(oppo_o_next_ep),
                            }
                        ])
                        returns_against_oppo_list[i][j].append(np.sum(agent_r_ep))     # 第i个对手策略与第j个受控代理策略每个episode交互得到的受控代理总回报
    
    num_trajs_list = []
    oppo_baseline_list = []
    oppo_target_list = []
    for i in range(num_oppo_policy):
        num_trajs_list.append(len(data_list[i]))
        returns_against_oppo_mean = []
        for j in range(num_age_policy):
            if returns_against_oppo_list[i][j] != []:
                returns_against_oppo_mean.append(np.mean(returns_against_oppo_list[i][j]))     # 对回报进行求均值
        oppo_baseline_list.append(np.mean(returns_against_oppo_mean))
        oppo_target_list.append(np.max(returns_against_oppo_mean))
    config_dict["NUM_TRAJS"] = num_trajs_list
    config_dict["OPPO_BASELINE"] = oppo_baseline_list
    config_dict["OPPO_TARGET"] = oppo_target_list
    LOG.info(f"num_trajs_list: {num_trajs_list}")
    LOG.info(f"oppo_baseline_list: {oppo_baseline_list}")
    LOG.info(f"oppo_baseline_mean: {np.mean(oppo_baseline_list)}")
    LOG.info(f"oppo_target_list: {oppo_target_list}")
    LOG.info(f"oppo_target_mean: {np.mean(oppo_target_list)}")
    return data_list


def cal_agent_oppo_obs_mean(trajectories, total=False):
    agent_total_obses, oppo_total_obses = [], []
    num_oppo_policy = len(trajectories)
    eps = 1e-6
    agent_obs_mean_list, oppo_obs_mean_list = [], []
    agent_obs_std_list, oppo_obs_std_list = [], []
    for i in range(num_oppo_policy):
        agent_obses_i, oppo_obses_i = [], []
        for traj in trajectories[i]:
            agent_total_obses.append(traj[0]['observations'])
            agent_obses_i.append(traj[0]['observations'])
            oppo_total_obses.append(traj[1]['observations'])
            oppo_obses_i.append(traj[1]['observations'])
        agent_obses_i = np.concatenate(agent_obses_i, axis=0)
        agent_obs_mean_list.append(np.mean(agent_obses_i, axis=0))
        agent_obs_std_list.append(np.std(agent_obses_i, axis=0) + eps)
        oppo_obses_i = np.concatenate(oppo_obses_i, axis=0)
        oppo_obs_mean_list.append(np.mean(oppo_obses_i, axis=0))
        oppo_obs_std_list.append(np.std(oppo_obses_i, axis=0) + eps)
    if total:
        agent_total_obses = np.concatenate(agent_total_obses, axis=0)
        agent_obs_mean_list = [np.mean(agent_total_obses, axis=0) for _ in range(num_oppo_policy)]
        agent_obs_std_list = [np.std(agent_total_obses, axis=0) + eps for _ in range(num_oppo_policy)]
        oppo_total_obses = np.concatenate(oppo_total_obses, axis=0)
        oppo_obs_mean_list = [np.mean(oppo_total_obses, axis=0) for _ in range(num_oppo_policy)]
        oppo_obs_std_list = [np.std(oppo_total_obses, axis=0) + eps for _ in range(num_oppo_policy)]
    return agent_obs_mean_list, agent_obs_std_list, oppo_obs_mean_list, oppo_obs_std_list


def online_episode_get_window(
        env,
        env_type,
        agent_obs_dim,
        oppo_obs_dim,
        act_dim,
        c_dim,
        encoder,
        decoder,
        oppo_policy,
        agent_index,
        oppo_index,
        num_steps=100,
        reward_scale=1000.,
        target_rtg=None,
        eval_mode='normal',
        agent_obs_mean=0.,
        agent_obs_std=1.,
        oppo_obs_mean=0.,
        oppo_obs_std=1.,
        oppo_context_window=None,
        device="cuda",
        obs_normalize=True,
):
    encoder.eval()
    encoder.to(device=device)
    K = encoder.K
    decoder.eval()
    decoder.to(device=device)

    agent_obs_mean = torch.from_numpy(agent_obs_mean).to(device=device)
    agent_obs_std = torch.from_numpy(agent_obs_std).to(device=device)
    oppo_obs_mean = torch.from_numpy(oppo_obs_mean).to(device=device)
    oppo_obs_std = torch.from_numpy(oppo_obs_std).to(device=device)

    if env_type == 'MS':
        time_step = env.reset()
        _, _, rel_state1, rel_state2 = get_two_state(time_step)
        obs_n = [rel_state1, rel_state2]
    else:
        obs_n = env.reset()
    if eval_mode == 'noise':
        for i in agent_index:
            obs_n[i] = obs_n[i] + np.random.normal(0, 0.1, size=obs_n[i].shape)

    if oppo_context_window != None:
        oppo_embeds, oppo_mask = [], []
        for oppo_trajs in oppo_context_window:
            n_o_oppo, a_oppo, r_oppo, _, _, timestep_oppo = oppo_trajs
            es = np.random.randint(0, n_o_oppo.shape[0])  # 返回一个随机整数
            oppo_embeds_, oppo_mask_ = encoder.get_tokens(
                n_o_oppo[es:es + K].to(device=device, dtype=torch.float32),
                a_oppo[es:es + K].to(device=device, dtype=torch.float32),
                r_oppo[es:es + K].to(device=device, dtype=torch.float32),
                timestep_oppo[es:es + K].to(device=device, dtype=torch.long),
                attention_mask=None,
            )    # 连续对手20个片段观察动作回报时间步
            oppo_embeds.append(oppo_embeds_)
            oppo_mask.append(oppo_mask_)
        oppo_embeds = torch.cat(oppo_embeds, dim=1).contiguous()
        oppo_mask = torch.cat(oppo_mask, dim=1).contiguous()
    else:
        oppo_embeds, oppo_mask = None, None

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_list_n = [None for _ in oppo_index + agent_index]
    act_list_n = [None for _ in oppo_index + agent_index]
    r_list_n = [None for _ in oppo_index + agent_index]
    target_rtg_list_n = [None for _ in oppo_index + agent_index]
    timestep_list_n = [None for _ in oppo_index + agent_index]
    for i in oppo_index + agent_index:
        obs_list_n[i] = torch.from_numpy(obs_n[i]).reshape(1, agent_obs_dim if i in agent_index else oppo_obs_dim).to(
            device=device, dtype=torch.float32)
        act_list_n[i] = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        r_list_n[i] = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = target_rtg
        target_rtg_list_n[i] = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timestep_list_n[i] = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# 对所有的o a r rtg timestep 初始化操作
    episode_return = [0. for _ in oppo_index + agent_index]
    true_steps = 0
    for t in range(num_steps):
        act_n = [None for _ in oppo_index + agent_index]
        for i in agent_index:
            # add padding
            act_list_n[i] = torch.cat([act_list_n[i], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[i] = torch.cat([r_list_n[i], torch.zeros(1, device=device)])
            if obs_normalize:
                action = decoder.get_action(
                    (obs_list_n[i].to(dtype=torch.float32) - agent_obs_mean) / agent_obs_std,
                    act_list_n[i].to(dtype=torch.float32),
                    r_list_n[i].to(dtype=torch.float32),
                    target_rtg_list_n[i].to(dtype=torch.float32),
                    timestep_list_n[i].to(dtype=torch.long),
                    oppo_embeds,
                    oppo_mask,
                )
            else:
                action = decoder.get_action(
                    obs_list_n[i].to(dtype=torch.float32),
                    act_list_n[i].to(dtype=torch.float32),
                    r_list_n[i].to(dtype=torch.float32),
                    target_rtg_list_n[i].to(dtype=torch.float32),
                    timestep_list_n[i].to(dtype=torch.long),
                    oppo_embeds,
                    oppo_mask,
                )
            if env_type == "PA":
                action = torch.nn.Softmax(dim=0)(action)
                action_index = torch.argmax(action)
                action = torch.eye(act_dim, dtype=torch.float32)[action_index]
                act = action.detach().clone().cpu().numpy()
                act = np.concatenate([act, np.zeros(c_dim)])
            elif env_type == 'MS':
                action_prob = torch.nn.Softmax(dim=0)(action)
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = torch.eye(act_dim, dtype=torch.float32)[act]
            act_n[i] = act
            act_list_n[i][-1] = action

        for j in oppo_index:
            act_list_n[j] = torch.cat([act_list_n[j], torch.zeros((1, act_dim), device=device)], dim=0)
            r_list_n[j] = torch.cat([r_list_n[j], torch.zeros(1, device=device)])
            if env_type == "PA":
                action = act = oppo_policy.action(obs_n[j])
            elif env_type == 'MS':
                action_prob = oppo_policy(torch.tensor(obs_n[j], dtype=torch.float32, device=device))
                dist = Categorical(action_prob)
                act = dist.sample().detach().clone().cpu().numpy()
                action = np.eye(act_dim, dtype=np.float32)[act]
            act_n[j] = act
            cur_action = torch.from_numpy(action[:act_dim]).to(device=device, dtype=torch.float32).reshape(1, act_dim)
            act_list_n[j][-1] = cur_action

        if env_type == 'MS':
            act_n = np.array(act_n)
            time_step = env.step(act_n)
            rew1, rew2 = time_step.rewards[0], time_step.rewards[1]
            reward_n = [rew1, rew2]
            _, _, rel_state1_, rel_state2_ = get_two_state(time_step)
            obs_n = [rel_state1_, rel_state2_]
            done_ = (time_step.last() == True)
            done_n = [done_, done_]
            info_n = {}
        else:
            obs_n, reward_n, done_n, info_n = env.step(act_n)
# 得到的实际的 o r done
        for i in oppo_index + agent_index:
            cur_obs = torch.from_numpy(obs_n[i]).to(device=device, dtype=torch.float32).reshape(1,agent_obs_dim if i in agent_index else oppo_obs_dim)
            obs_list_n[i] = torch.cat([obs_list_n[i], cur_obs], dim=0)
            r_list_n[i][-1] = reward_n[i]

            if eval_mode != 'delayed':
                pred_return = target_rtg_list_n[i][0, -1] - (reward_n[i] / reward_scale)
            else:
                pred_return = target_rtg_list_n[i][0, -1]
            target_rtg_list_n[i] = torch.cat(
                [target_rtg_list_n[i], pred_return.reshape(1, 1)], dim=1)
            timestep_list_n[i] = torch.cat(
                [timestep_list_n[i],
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
            episode_return[i] += reward_n[i]
        true_steps += 1
        if done_n[0] or done_n[1]:
            break

    for j in oppo_index:
        if obs_normalize:
            n_o_oppo = (obs_list_n[j] - oppo_obs_mean) / oppo_obs_std
        else:
            n_o_oppo = obs_list_n[j]
        a_oppo = act_list_n[j]
        r_oppo = r_list_n[j]
        timestep_oppo = timestep_list_n[j]
    # for i in agent_index:
    #     if obs_normalize:
    #         n_o_agent = (obs_list_n[i] - agent_obs_mean) / agent_obs_std
    #     else:
    #         n_o_agent = obs_list_n[i]
    #     a_agent = act_list_n[i]
    #     r_agent = r_list_n[i]
    #     timestep_agent = timestep_list_n[i]

    steps_ = min(num_steps, true_steps)
    if oppo_context_window == None:
        oppo_context_window = [(n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1])]
    else:
        oppo_context_window.append((n_o_oppo[1:1 + steps_, :], a_oppo[:steps_, :], r_oppo[:steps_], n_o_oppo[:steps_, :],
                                   timestep_oppo[0, :steps_], timestep_oppo[0, 1:steps_ + 1]))

    average_epi_return = np.mean([episode_return[k] for k in agent_index])

    return average_epi_return, oppo_context_window
