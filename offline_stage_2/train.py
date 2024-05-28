import os, sys
from pprint import PrettyPrinter
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import time
from offline_stage_1.net import GPTEncoder
from offline_stage_1.utils import CrossEntropy
from offline_stage_2.config import Config, get_config_dict
from offline_stage_2.net import GPTDecoder
from offline_stage_2.nn_trainer import ResponsePolicyTrainer
from offline_stage_2.utils import (
    load_agent_oppo_data,
    cal_agent_oppo_obs_mean,
    get_batch,
    eval_episodes,
    get_env_and_oppo,
    LOG,
)
import torch
import numpy as np
import wandb
if Config.RUN_OFFLINE:
    os.environ["WANDB_MODE"] = "offline"


def main():

    env_type = Config.ENV_TYPE
    agent_obs_dim = Config.AGENT_OBS_DIM
    oppo_obs_dim = Config.OPPO_OBS_DIM
    act_dim = Config.ACT_DIM
    num_steps = Config.NUM_STEPS
    K_decoder = Config.K   # ？？？ k_decoder?
    obs_normalize = Config.OBS_NORMALIZE
    average_total_obs = Config.AVERAGE_TOTAL_OBS
    
    exp_id = Config.EXP_ID
    log_to_wandb = Config.WANDB
    
    device = Config.DEVICE
    num_iter = Config.NUM_ITER
    num_update_per_iter = Config.NUM_UPDATE_PER_ITER
    checkpoint_freq = Config.CHECKPOINT_FREQ
    seen_eval_interval = Config.SEEN_EVAL_INTERVAL
    unseen_eval_interval = Config.UNSEEN_EVAL_INTERVAL
    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE
    warmup_steps = Config.WARMUP_STEPS
    weight_decay = Config.WEIGHT_DECAY
    
    hidden_dim = Config.HIDDEN_DIM
    dropout = Config.DROPOUT
    num_layer = Config.NUM_LAYER
    num_head = Config.NUM_HEAD
    activation_func = Config.ACTIVATION_FUNC
    action_tanh = Config.ACTION_TANH
    
    agent_index = Config.AGENT_INDEX
    oppo_index = Config.OPPO_INDEX
    
    data_path = Config.OFFLINE_DATA_PATH
    save_model_dir = Config.MODEL_DIR
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir, exist_ok=False)
    
    seen_oppo_policy = Config.SEEN_OPPO_POLICY
    LOG.info(f'Seen opponent policy list: {seen_oppo_policy}')
    unseen_oppo_policy = Config.UNSEEN_OPPO_POLICY
    LOG.info(f'Unseen opponent policy list: {unseen_oppo_policy}')
    
    CONFIG_DICT = get_config_dict()
    seed = CONFIG_DICT["SEED_RES"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env_and_seen_oppo = get_env_and_oppo(CONFIG_DICT, seen_oppo_policy)
    env_and_unseen_oppo = get_env_and_oppo(CONFIG_DICT, unseen_oppo_policy)
    
    offline_data = load_agent_oppo_data(data_path, agent_index, oppo_index, act_dim, CONFIG_DICT)
    LOG.info("Finish loading offline dataset.")

    if obs_normalize:
        agent_obs_mean_list, agent_obs_std_list, oppo_obs_mean_list, oppo_obs_std_list = cal_agent_oppo_obs_mean(offline_data, total=average_total_obs)
        CONFIG_DICT["AGENT_OBS_MEAN"] = agent_obs_mean_list
        CONFIG_DICT["AGENT_OBS_STD"] = agent_obs_std_list
        CONFIG_DICT["OPPO_OBS_MEAN"] = oppo_obs_mean_list
        CONFIG_DICT["OPPO_OBS_STD"] = oppo_obs_std_list
    
    exp_prefix = env_type
    num_oppo_policy = CONFIG_DICT["NUM_OPPO_POLICY"]
    group_name = f'{exp_prefix}-{num_oppo_policy}oppo'
    curtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_prefix = f'{group_name}-{exp_id}-{curtime}'
    LOG.info("--------------------- EXP INFO ---------------------")
    PrettyPrinter().pprint(CONFIG_DICT)
    LOG.info("----------------------------------------------------")
    
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
    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    encoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        encoder_optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    decoder = GPTDecoder(
        conf=CONFIG_DICT,
        obs_dim=agent_obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_dim,
        max_length=K_decoder,       # 最大长度 ？？
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
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    decoder_scheduler = torch.optim.lr_scheduler.LambdaLR(
        decoder_optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    
    trainer = ResponsePolicyTrainer(
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        encoder_scheduler=encoder_scheduler,
        decoder_scheduler=decoder_scheduler,
        get_batch_fn=get_batch(offline_data, CONFIG_DICT),
        loss_fn=CrossEntropy,
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
    
    LOG.info("Start opponent-aware response policy training.")
    for i in range(num_iter):
        outputs = trainer.train(
            num_update=num_update_per_iter,
        )
        
        if i % checkpoint_freq == 0 or i == num_iter-1:
            trainer.save_model(
                postfix=f"_iter_{i}",
                save_dir=save_model_dir,
            )
            LOG.info(f"Finish training of iteration [{i}].")
        
        if i % seen_eval_interval == 0 or i == num_iter-1:
            seen_eval_logs = trainer.eval(
                eval_episodes,
                seen_oppo_policy,
                env_and_seen_oppo,
                eval_type="seen",
            )
            outputs.update(seen_eval_logs)
            LOG.info(f'Evaluation result of iteration [{i}]:')
            for k, v in seen_eval_logs.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)
        
        if i % unseen_eval_interval == 0 or i == num_iter-1:
            unseen_eval_logs = trainer.eval(
                eval_episodes,
                unseen_oppo_policy,
                env_and_unseen_oppo,
                eval_type="unseen",
            )
            outputs.update(unseen_eval_logs)
            LOG.info(f'Evaluation result of iteration [{i}]:')
            for k, v in unseen_eval_logs.items():
                LOG.info(f'{k}: {v}')
            LOG.info('=' * 80)

        outputs.update({"global_step": i})
        
        if log_to_wandb:
            wandb.log(outputs)
    
    LOG.info(f"Finish training opponent-aware response policy.")


if __name__ == '__main__':
    main()