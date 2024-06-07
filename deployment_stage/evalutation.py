import os, sys
from pprint import PrettyPrinter
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
from offline_stage_1.config import Config, get_config_dict
from offline_stage_1.net import GPTEncoder
from offline_stage_1.nn_trainer import PolicyEmbeddingTrainer
from offline_stage_1.utils import (
    load_oppo_data,
    cal_obs_mean,
    CrossEntropy,
    LOG,
)
from deployment_stage.utils import (
    evaluation_online_data,
    evaluation,
)
import torch
import numpy as np
import wandb
from deployment_stage.collect_data import collect_data
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"


def main():
    env_type = Config.ENV_TYPE  # 什么数据集（环境）
    obs_dim = Config.OBS_DIM  # o 对手的观察向量维度
    act_dim = Config.ACT_DIM  # a 对手的动作向量维度
    num_steps = Config.NUM_STEPS  # 每个 episode 的最大步骤数

    exp_id = Config.EXP_ID
    seed = Config.SEED_PEL
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_to_wandb = Config.WANDB
    device = Config.DEVICE
    obs_normalize = Config.OBS_NORMALIZE  # 对对手的观察是否标准化
    average_total_obs = Config.AVERAGE_TOTAL_OBS  # 对对手的观察的总数是否平均？？？
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER  # 三层
    num_head = Config.NUM_HEAD  # 单头自注意力
    activation_func = Config.ACTIVATION_FUNC

    hidden_dim = Config.HIDDEN_DIM

    agent_index = Config.AGENT_INDEX  # 受控代理的index
    oppo_index = Config.OPPO_INDEX  # 对手的 index

    data_path = Config.OFFLINE_DATA_PATH
    save_model_dir = Config.PEL_MODEL_DIR

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir, exist_ok=False)

    CONFIG_DICT = get_config_dict()  # 离线阶段1的 配置字典

    offline_data = load_oppo_data(data_path, oppo_index, act_dim, CONFIG_DICT)  # 将数据集 对手索引，动作维度，一系列config  加载对手数据集
    LOG.info("Finish loading offline dataset.")

    if obs_normalize:
        obs_offline_mean_list, obs_offline_std_list = cal_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["offline_OBS_MEAN"] = obs_offline_mean_list
        CONFIG_DICT["offline_OBS_STD"] = obs_offline_std_list
    online_data = collect_data()
    LOG.info("Finish loading online dataset.")
    if obs_normalize:
        obs_online_mean_list, obs_online_std_list = cal_obs_mean(online_data, total=average_total_obs)
        CONFIG_DICT["online_OBS_MEAN"] = obs_online_mean_list
        CONFIG_DICT["online_OBS_STD"] = obs_online_std_list

    exp_prefix = env_type
    num_oppo_policy = len(offline_data)+len(online_data)
    group_name = f'{exp_prefix}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    LOG.info("--------------------- EXP INFO ---------------------")
    PrettyPrinter().pprint(CONFIG_DICT)
    LOG.info("----------------------------------------------------")

    encoder = GPTEncoder(
        conf=CONFIG_DICT,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_ep_len=(num_steps + 20),
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
    encoder.load_model("../offline_stage_2/model/PA-pretrained_models/res_encoder_iter_1999", device=device)
    encoder.eval()

    all_online_data = evaluation_online_data(online_data,CONFIG_DICT)

    evaluation(
        encoder=encoder,
        all_data=all_online_data,
        config=CONFIG_DICT,
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=CONFIG_DICT["PROJECT_NAME"],
            config=CONFIG_DICT,
        )
        save_model_dir += wandb.run.name
        os.mkdir(save_model_dir)


if __name__ == '__main__':
    main()

