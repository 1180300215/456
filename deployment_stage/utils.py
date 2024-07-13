import logging
import time
import numpy as np
import torch
from torch import nn
import wandb
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_2.utils import eval_episode_rtg
from offline_stage_2.utils import evaluating_online_rtg
from offline_stage_2.utils import online_episode_get_window
from offline_stage_2.utils import online_get_oppo_agent_window
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
LOG = logging.getLogger()


def test_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config, args, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = args.device
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    oppo_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg = config["OPPO_TARGET"][oppo_id]
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            # else:
            #     target_rtg = np.mean(config["OPPO_TARGET"])
            #     target_rtg = 1.5
            #     if config['OBS_NORMALIZE']:
            #         agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
            #         oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
            #     else:
            #         agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
            #         oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)

            elif oppo_name == config["UNSEEN_OPPO_POLICY"][0]:
                target_rtg = 70
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            
            elif oppo_name == config["UNSEEN_OPPO_POLICY"][1]:
                target_rtg = np.mean(config["OPPO_TARGET"])
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)

            else:  
                target_rtg = 52              
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            
            encoder_path = ['../offline_stage_2/model/PA-pretrained_models/res_encoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240630032440/res_encoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240703071700/res_encoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240704062739/res_encoder_iter_1999']
            decoder_path = ['../offline_stage_2/model/PA-pretrained_models/res_decoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240630032440/res_decoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240703071700/res_decoder_iter_1999','model/PA-6oppo-ours-a1-l1-W5-K20-20240704062739/res_decoder_iter_1999']
            # encoder_path = ['../offline_stage_2/model/MS-pretrained_models/res_encoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240709062744/res_encoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240709065011/res_encoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240710015153/res_encoder_iter_1999']
            # decoder_path = ['../offline_stage_2/model/MS-pretrained_models/res_decoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240709062744/res_decoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240709065011/res_decoder_iter_1999','model/MS-6oppo-ours-a1-l1-W5-K20-20240710015153/res_decoder_iter_1999']
            return_eval = [ _ for _ in range(len(encoder_path))]
            for j in range(len(encoder_path)):
                encoder.load_model(encoder_path[j],device=device)
                encoder.eval()
                decoder.load_model(decoder_path[j],device=device)
                decoder.eval()
                return_all = []
                oppo_evaluate_context_window = None
                for k in range(30):
                    with torch.no_grad():
                        ret, oppo_evaluate_context_window_new = eval_episode_rtg(
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
                            oppo_context_window=oppo_evaluate_context_window,
                            device=device,
                            obs_normalize=config['OBS_NORMALIZE'],
                            )
                        oppo_evaluate_context_window = oppo_evaluate_context_window_new
                        oppo_evaluate_context_window = oppo_evaluate_context_window[-ocw_size:]
                    return_all.append(ret)
                return_eval[j] = np.mean(return_all[-30:])
            eval_index = 0
            max_eval = return_eval[0]
            for j in range(len(encoder_path)):
                if return_eval[j]>max_eval :
                    eval_index = j
                    max_eval= return_eval[j]
            cur_encoder_path = encoder_path[eval_index]
            cur_decoder_path = decoder_path[eval_index]
            # print(cur_encoder_path)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        encoder.load_model(cur_encoder_path,device=device)
        encoder.eval()
        decoder.load_model(cur_decoder_path,device=device)
        decoder.eval()
    
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
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i+1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})


def collect_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config,log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = 'cuda:0'
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    oppo_context_window = None
    oppo_part_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0] + '_' + oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg = config["OPPO_TARGET"][oppo_id]
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                target_rtg = 1.0
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0),axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        with torch.no_grad():
            ret, oppo_context_window_new = online_episode_get_window(
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
                oppo_part_context_window=oppo_part_context_window,
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
            )
            oppo_context_window = oppo_context_window_new
            oppo_part_context_window = oppo_context_window[-ocw_size:]
            # oppo_context_window = oppo_context_window[-ocw_size:]
        returns.append(ret)
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i + 1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    length = len(oppo_context_window)
    LOG.info(f'ocw的长度为: {length}')
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    return oppo_context_window


def evaluation_online_data(online_data, config_dict):
    obs_dim = config_dict["OBS_DIM"]
    act_dim = config_dict["ACT_DIM"]
    K = config_dict["NUM_STEPS"]  # 最大值 每个episode
    if config_dict["OBS_NORMALIZE"]:
        obs_online_mean_list = config_dict["online_OBS_MEAN"]
        obs_online_std_list = config_dict["online_OBS_STD"]
    device = config_dict["DEVICE"]
    online_all_data = [oppo_data for oppo_data_list in online_data for oppo_data in oppo_data_list]

    def fn(batch_size=100, max_len=K):
        n_o, a, r, timesteps, mask = [], [], [], [], []
        for k in range(batch_size):
            traj = online_all_data[k]

            n_o.append(traj['next_observations'][:max_len].reshape(1, -1, obs_dim))  # 加入一个 1*100*8
            a.append(traj['actions'][:max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][:max_len].reshape(1, -1, 1))
            timesteps.append(
                np.arange(1, (n_o[-1].shape[1] + 1)).reshape(1, -1)
            )  # 1- 100
            timesteps[-1][timesteps[-1] >= (max_len + 1)] = max_len
            tlen = n_o[-1].shape[1]
            n_o[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), n_o[-1]], axis=1)
            if config_dict["OBS_NORMALIZE"]:
                obs_mean, obs_std = obs_online_mean_list[0], obs_online_std_list[0]
                n_o[-1] = (n_o[-1] - obs_mean) / obs_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)

            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        n_o = torch.from_numpy(np.concatenate(n_o, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return n_o, a, r, timesteps, mask
    return fn


def evaluation(encoder, all_data,config):
    # print("niubi")
    avg_pooling = nn.AvgPool1d(kernel_size=config["NUM_STEPS"])
    batch = all_data()
    n_o, a, r, timesteps, mask = batch
    hidden_states = encoder(
        obs=n_o,
        action=a,
        reward=r,
        timestep=timesteps,
        attention_mask=mask,
    )

    hidden_states = avg_pooling(hidden_states.permute([0, 2, 1])).squeeze(-1)

    hidden_mean = hidden_states.mean(dim=0)
    hidden_std = hidden_states.std(dim=0)

    print("当前环境下均值为：")
    print(hidden_mean)
    print("当前环境下方差为：")
    print(hidden_std)


def load_online_data(oppo_context_w):
    online_data_list = [[]]
    online_cw_size = len(oppo_context_w)
    for i in range(online_cw_size):
        num_steps = len(oppo_context_w[i][0])
        o_ep = []
        a_ep = []
        r_ep = []
        o_next_ep = []
        for j in range(num_steps):
            o_ep.append(np.array(oppo_context_w[i][3][j].cpu()))
            a_ep.append(np.array(oppo_context_w[i][1][j].cpu()))
            r_ep.append(np.array(oppo_context_w[i][2][j].cpu()))
            o_next_ep.append(np.array(oppo_context_w[i][0][j].cpu()))
        online_data_list[0].append(
            {
                "observations": np.array(o_ep),
                "actions": np.array(a_ep),
                "rewards": np.array(r_ep),
                "next_observations": np.array(o_next_ep),
            }
        )
    return online_data_list


def evaluating_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config, args, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = args.device
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    oppo_context_window = None
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0]+'_'+oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg = config["OPPO_TARGET"][oppo_id]
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                target_rtg = np.mean(config["OPPO_TARGET"])
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0), axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        with torch.no_grad():
            ret, oppo_context_window_new = evaluating_online_rtg(
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
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i+1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})



    length = len(all_context_window)
    LOG.info(f'ocw的长度为: {length}')
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    return all_context_window


def load_all_data(all_context_window):
    all_data_list = [[]]
    all_data_size = len(all_context_window)
    for i in range(all_data_size):
        num_steps = 100
        agent_o_ep = []
        agent_a_ep = []
        agent_r_ep = []
        oppo_o_ep = []
        oppo_a_ep = []
        oppo_r_ep = []
        oppo_o_next_ep = []
        for j in range(num_steps):
            agent_o_ep.append(np.array(all_context_window[i][3][j].cpu()))
            agent_a_ep.append(np.array(all_context_window[i][1][j].cpu()))
            agent_r_ep.append(np.array(all_context_window[i][2][j].cpu()))
            oppo_a_ep.append(np.array(all_context_window[i][7][j].cpu()))
            oppo_o_ep.append(np.array(all_context_window[i][9][j].cpu()))
            oppo_r_ep.append(np.array(all_context_window[i][8][j].cpu()))
            oppo_o_next_ep.append(np.array(all_context_window[i][6][j].cpu()))
        all_data_list[0].append([
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
    return all_data_list


def collect_all_episode(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config, log_to_wandb):
    LOG.info(f'Testing against opponent policies: {test_oppo_policy}')
    LOG.info(f'# of total testing episodes: {num_test}')
    LOG.info(f'# of total testing opponent policies: {num_test // switch_interval}')
    env = env_and_test_oppo["env"]
    agent_obs_dim, oppo_obs_dim, act_dim = config['AGENT_OBS_DIM'], config['OPPO_OBS_DIM'], config['ACT_DIM']
    agent_index, oppo_index = config["AGENT_INDEX"], config["OPPO_INDEX"]
    num_steps = config["NUM_STEPS"]
    env_type = config["ENV_TYPE"]
    c_dim = config["C_DIM"]
    reward_scale = config["REWARD_SCALE"]
    device = 'cuda:2'
    eval_mode = config['EVAL_MODE']
    test_oppo_indexes = config["TEST_OPPO_INDEXES"]
    ocw_size = config["OCW_SIZE"]
    encoder.eval()
    decoder.eval()

    returns = []
    cur_test_oppo_index = 0
    all_context_window = None
    part_context_window = None  #  change
    for i in range(num_test):
        outputs = dict()
        if i % switch_interval == 0:
            test_start = time.time()
            oppo_id = test_oppo_indexes[cur_test_oppo_index]
            oppo_name = test_oppo_policy[oppo_id]
            if isinstance(oppo_name, tuple):
                oppo_name_ = oppo_name[0] + '_' + oppo_name[-1]
            else:
                oppo_name_ = oppo_name
            oppo_policy = env_and_test_oppo["oppo_policy"][oppo_id]
            if oppo_name in config["SEEN_OPPO_POLICY"]:
                target_rtg = config["OPPO_TARGET"][oppo_id]
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = config['AGENT_OBS_MEAN'][oppo_id], config['AGENT_OBS_STD'][oppo_id]
                    oppo_obs_mean, oppo_obs_std = config['OPPO_OBS_MEAN'][oppo_id], config['OPPO_OBS_STD'][oppo_id]
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            else:
                target_rtg = 1.0
                # np.mean(config["OPPO_TARGET"])
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0),axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0),axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
                else:
                    agent_obs_mean, agent_obs_std = np.array(0.), np.array(1.)
                    oppo_obs_mean, oppo_obs_std = np.array(0.), np.array(1.)
            LOG.info(f'Start testing against opponent policies: {oppo_name_} ...')

        with torch.no_grad():
            ret, all_context_window_new = online_get_oppo_agent_window(
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
                all_context_window=all_context_window,
                part_context_window=part_context_window,
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
            )
            all_context_window = all_context_window_new
            part_context_window = all_context_window_new[-ocw_size:]
        returns.append(ret)
        outputs.update({
            'test-epi/global_return': ret,
            f'test-epi/{oppo_name_}_target_{target_rtg:.3f}_return': ret,
            "granularity/num_episode": i,
        })
        if (i + 1) % switch_interval == 0:
            test_oppo_log = {
                'test-oppo/oppo_return': np.mean(returns[-switch_interval:]),
                "granularity/num_opponent_policy": cur_test_oppo_index,
                "time/testing": time.time() - test_start,
            }
            outputs.update(test_oppo_log)
            LOG.info(f'Testing result of opponent [{cur_test_oppo_index}]:')
            for k, v in test_oppo_log.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
            cur_test_oppo_index += 1
        if log_to_wandb:
            wandb.log(outputs)
    return_mean = np.mean(returns)
    length = len(all_context_window)
    LOG.info(f'ocw的长度为: {length}')
    LOG.info(f'Average return against all opponent policies: {return_mean}')
    if log_to_wandb:
        wandb.log({'test-epi/global_return_mean': return_mean})
    return all_context_window


def load_all_data(all_context_window):
    all_data_list = [[]]
    all_data_size = len(all_context_window)
    for i in range(all_data_size):
        num_steps = len(all_context_window[i][0])
        agent_o_ep = []
        agent_a_ep = []
        agent_r_ep = []
        oppo_o_ep = []
        oppo_a_ep = []
        oppo_r_ep = []
        oppo_o_next_ep = []
        for j in range(num_steps):
            agent_o_ep.append(np.array(all_context_window[i][3][j].cpu()))
            agent_a_ep.append(np.array(all_context_window[i][1][j].cpu()))
            agent_r_ep.append(np.array(all_context_window[i][2][j].cpu()))
            oppo_a_ep.append(np.array(all_context_window[i][7][j].cpu()))
            oppo_o_ep.append(np.array(all_context_window[i][9][j].cpu()))
            oppo_r_ep.append(np.array(all_context_window[i][8][j].cpu()))
            oppo_o_next_ep.append(np.array(all_context_window[i][6][j].cpu()))
        all_data_list[0].append([
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
    return all_data_list
