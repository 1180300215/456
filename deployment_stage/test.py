import argparse
import os, sys
import pickle
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
from offline_stage_1.net import GPTEncoder
from offline_stage_2.net import GPTDecoder
from offline_stage_2.config import Config, get_config_dict
from offline_stage_2.utils import (
    cal_agent_oppo_obs_mean,
    get_env_and_oppo,
    load_agent_oppo_data,
)
from deployment_stage.utils import (
    test_episodes,
    LOG,
)
import torch
import numpy as np
import wandb
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"

def main(args):
    env_type = Config.ENV_TYPE
    agent_obs_dim = Config.AGENT_OBS_DIM
    oppo_obs_dim = Config.OPPO_OBS_DIM
    act_dim = Config.ACT_DIM
    num_steps = Config.NUM_STEPS
    K_decoder = Config.K
    obs_normalize = Config.OBS_NORMALIZE
    average_total_obs = Config.AVERAGE_TOTAL_OBS
    
    exp_id = Config.EXP_ID
    
    log_to_wandb = Config.WANDB
    
    device = args.device
    test_mode = args.test_mode
    num_test = args.num_test
    switch_interval = args.switch_interval
    hidden_dim = Config.HIDDEN_DIM
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER
    num_head = Config.NUM_HEAD
    activation_func = Config.ACTIVATION_FUNC
    action_tanh = Config.ACTION_TANH
    
    agent_index = Config.AGENT_INDEX
    oppo_index = Config.OPPO_INDEX
    
    decoder_path = args.decoder_param_path
    data_path = Config.OFFLINE_DATA_PATH
    
    seen_oppo_policy = Config.SEEN_OPPO_POLICY
    unseen_oppo_policy = Config.UNSEEN_OPPO_POLICY
    
    CONFIG_DICT = get_config_dict()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    CONFIG_DICT["SEED_RES"] = seed
    CONFIG_DICT["DEVICE"] = device
    CONFIG_DICT["EVAL_DEVICE"] = device
    
    if test_mode == "seen":
        test_oppo_policy = seen_oppo_policy
    elif test_mode == "unseen":
        test_oppo_policy = unseen_oppo_policy
    elif test_mode == "mix":
        test_oppo_policy = seen_oppo_policy+unseen_oppo_policy
    
    env_and_test_oppo = get_env_and_oppo(CONFIG_DICT, test_oppo_policy)
    
    offline_data = load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, config_dict=CONFIG_DICT)
    LOG.info("Finish loading offline dataset.")

    if obs_normalize:
        agent_obs_mean_list, agent_obs_std_list, oppo_obs_mean_list, oppo_obs_std_list = cal_agent_oppo_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["AGENT_OBS_MEAN"] = agent_obs_mean_list
        CONFIG_DICT["AGENT_OBS_STD"] = agent_obs_std_list
        CONFIG_DICT["OPPO_OBS_MEAN"] = oppo_obs_mean_list
        CONFIG_DICT["OPPO_OBS_STD"] = oppo_obs_std_list
    

    with open(f"utility/{test_mode}_oppo_indexes.npy", 'rb') as f:
        test_oppo_indexes = np.load(f)
        CONFIG_DICT["TEST_OPPO_INDEXES"] = test_oppo_indexes
        LOG.info(f"{test_mode}_oppo_indexes: {test_oppo_indexes}")
    
    exp_prefix = env_type
    num_oppo_policy = len(test_oppo_policy)
    group_name = f'{exp_prefix}-{test_mode}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    
    encoder = GPTEncoder(
        conf=CONFIG_DICT,
        obs_dim=oppo_obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_ep_len=(num_steps+20),
        activation_function=activation_func,
        n_layer=num_layer,
        n_head=num_head,
        n_inner=4 * hidden_dim,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        add_cross_attention=False,
    )
    encoder = encoder.to(device=device)
    # encoder.load_model(args.encoder_param_path, device=device)
    encoder.eval()
    
    decoder = GPTDecoder(
        conf=CONFIG_DICT,
        obs_dim=agent_obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_length=K_decoder,
        max_ep_len=(num_steps+20),
        activation_function=activation_func,
        n_layer=num_layer,
        n_head=num_head,
        n_inner=4 * hidden_dim,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
        action_tanh=action_tanh,
        add_cross_attention=True,
    )
    decoder = decoder.to(device=device)
    # decoder.load_model(decoder_path, device=device)
    decoder.eval()
    
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=args.project_name + f"-{test_mode}",
            config=CONFIG_DICT,
        )
    
    LOG.info("Start testing TAO.")
    LOG.info(f'Testing mode: {test_mode}')
    test_episodes(
        encoder=encoder,
        decoder=decoder,
        env_and_test_oppo=env_and_test_oppo,
        num_test=num_test,
        switch_interval=switch_interval,
        test_oppo_policy=test_oppo_policy,
        config=CONFIG_DICT,
        args=args,
        log_to_wandb=log_to_wandb,
    )
    LOG.info(f"Finish testing TAO.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ---------- NOTE: TAO testing ----------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:2')
    # parser.add_argument("--project_name", type=str, default="TEST-MS")
    # * remember to change ENV_TYPE in '../offline_stage_2/config.py' file to PA when testing on PA
    parser.add_argument("--project_name", type=str, default="TEST-PA-evaluate")
    parser.add_argument("--num_test", type=int, default=2500)
    parser.add_argument("--switch_interval", type=int, default=50)
    parser.add_argument("--test_mode", type=str, default="mix", choices=["seen", "unseen", "mix"])
    args = parser.parse_args()
    
    if "MS" in args.project_name:  # * MS
        args.encoder_param_path = '../offline_stage_2/model/MS-pretrained_models/res_encoder_iter_1999'
        args.decoder_param_path = '../offline_stage_2/model/MS-pretrained_models/res_decoder_iter_1999'
    elif "PA" in args.project_name:  # * PA
        args.encoder_param_path = '../offline_stage_2/model/PA-pretrained_models/res_encoder_iter_1999'   
        args.decoder_param_path = '../offline_stage_2/model/PA-pretrained_models/res_decoder_iter_1999'
    
    main(args)