import logging
import time
import numpy as np
import torch
import wandb
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_2.utils import eval_episode_rtg
from offline_stage_2.utils import online_episode_get_window
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


def collect_episodes(encoder, decoder, env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config, args,log_to_wandb):
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
                target_rtg = np.mean(config["OPPO_TARGET"])
                if config['OBS_NORMALIZE']:
                    agent_obs_mean, agent_obs_std = np.mean(np.stack(config['AGENT_OBS_MEAN'], axis=0),
                                                            axis=0), np.mean(np.stack(config['AGENT_OBS_STD'], axis=0),
                                                                             axis=0)
                    oppo_obs_mean, oppo_obs_std = np.mean(np.stack(config['OPPO_OBS_MEAN'], axis=0), axis=0), np.mean(
                        np.stack(config['OPPO_OBS_STD'], axis=0), axis=0)
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
                device=device,
                obs_normalize=config['OBS_NORMALIZE'],
            )
            oppo_context_window = oppo_context_window_new
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



def train_episodes(encoder, decoder, oppo_context_w,env_and_test_oppo, num_test, switch_interval, test_oppo_policy, config, args, log_to_wandb):
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


def load_online_data(oppo_context_w):
    online_cw_size = len(oppo_context_w)
    for i in range(online_cw_size):
        num_steps = len(online_cw_size[i][0])
        o_ep = []
        a_ep = []
        r_ep = []
        o_next_ep = []
        for j in range (num_steps):




