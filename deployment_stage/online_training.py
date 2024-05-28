# import argparse
# import os, sys
# from collect_data import main
#
# collected_data = main()
#
# for _ in range(batch_num):

#
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
# import time
# from offline_stage_1.net import GPTEncoder
# from offline_stage_2.net import GPTDecoder
# from offline_stage_2.config import Config, get_config_dict
# from offline_stage_2.utils import (
#     cal_agent_oppo_obs_mean,
#     get_env_and_oppo,
#     load_agent_oppo_data,
# )
# from deployment_stage.utils import (
#     collect_episodes,
#     train_episodes,
#     LOG,
# )
# from deployment_stage.collect_data import (
#
# )
# import torch
# import numpy as np
# import wandb
#
# if Config.RUN_OFFLINE:
#     os.environ["WANDB_MODE"] = "offline"
#
