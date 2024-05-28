
class Config:
    WANDB = True
    RUN_OFFLINE = False

    # PROJECT_NAME = 'RES-PA'
    PROJECT_NAME = 'RES-MS'
    
    # ENV_TYPE = "PA"
    ENV_TYPE = "MS"
    
    # ---------- NOTE: environment settings ----------
    if ENV_TYPE == "PA":  # * PA
        SCENARIO = "simple_adversary.py"
        OPPO_OBS_DIM = 8  # opponent obs_dim
        AGENT_OBS_DIM = 10  # agent obs_dim
        ACT_DIM = 5
        C_DIM = 2    # ？？？
        NUM_STEPS = 100
        K = 20   # ？？？
        REWARD_SCALE = 100
    elif ENV_TYPE == "MS":  # * MS
        SCENARIO = 'markov_soccer'
        OPPO_OBS_DIM = 12  # opponent obs_dim
        AGENT_OBS_DIM = 12  # agent obs_dim
        ACT_DIM = 5
        C_DIM = 0
        NUM_STEPS = 100
        K = 20
        REWARD_SCALE = 1
    
    OBS_NORMALIZE = True
    AVERAGE_TOTAL_OBS = True
    
    # ---------- NOTE: opponent settings ----------
    if ENV_TYPE == "PA":  # * PA
        OPPO_INDEX = [0]
        AGENT_INDEX = [1,2]
        SEEN_OPPO_POLICY = ["FixOnePolicy", "ChaseOnePolicy", "MiddlePolicy", "BouncePolicy", ("RLPolicy", "opponent_policy_models/PPO_5000.pt", "5")]
        UNSEEN_OPPO_POLICY = ["FixThreePolicy", "ChaseBouncePolicy", ("RLPolicy", "opponent_policy_models/PPO_10000.pt", "6")]
        OFFLINE_DATA_PATH = '../envs/multiagent_particle_envs/data/offline_dataset_PA_5oppo_10k.pkl'
    elif ENV_TYPE == "MS":  # * MS
        OPPO_INDEX = [0]
        AGENT_INDEX = [1]
        SEEN_OPPO_POLICY = ['SnatchAttackPolicy', 'SnatchEvadePolicy', ('RLPolicy', 'opponent_policy_models/TRCoPO_30000.pth', '1'), ('RLPolicy', 'opponent_policy_models/TRGDA_10000.pth', '2'), ('RLPolicy', 'opponent_policy_models/PPO_50000.pt', '3'),]
        UNSEEN_OPPO_POLICY = ['GuardAttackPolicy', 'GuardEvadePolicy', ('RLPolicy', 'opponent_policy_models/PPO_100000.pt', '4'),]
        OFFLINE_DATA_PATH = '../envs/markov_soccer/data/offline_dataset_MS_5oppo_10k.pkl'
    
    
    # ---------- NOTE: global hyper-parameters ----------
    SEED_RES = 0
    EXP_ID = 'ours-a1-l1'
    DEVICE = 'cuda:0'
    EVAL_DEVICE = DEVICE
    
    OCW_SIZE = 5
    
    if ENV_TYPE == "PA":  # * PA
        ENCODER_PARAM_PATH = '../offline_stage_1/model/PA-pretrained_models/pel_encoder_iter_499'
    elif ENV_TYPE == "MS":  # * MS
        ENCODER_PARAM_PATH = '../offline_stage_1/model/MS-pretrained_models/pel_encoder_iter_199'
    
    EXP_ID += f'-W{OCW_SIZE}' if OCW_SIZE is not None else ''
    EXP_ID += f'-K{K}'
    
    
    # ---------- NOTE: opponent-aware response policy training ----------
    if ENV_TYPE == 'PA':  # * PA
        NUM_ITER = 2000
        BATCH_SIZE = 128
    elif ENV_TYPE == 'MS':  # * MS
        NUM_ITER = 2000
        BATCH_SIZE = 128
    NUM_UPDATE_PER_ITER = 10
    CHECKPOINT_FREQ = 200
    WARMUP_STEPS = 10000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD = 0.5
    
    # ---------- NOTE: evaluation during training ----------
    if ENV_TYPE == 'PA':  # * PA
        NUM_EVAL_EPISODES = 100
    elif ENV_TYPE == 'MS':  # * MS
        NUM_EVAL_EPISODES = 100
    SEEN_EVAL_INTERVAL = CHECKPOINT_FREQ
    UNSEEN_EVAL_INTERVAL = CHECKPOINT_FREQ
    EVAL_MODE = "normal"
    
    # ---------- NOTE: dir for training ----------
    MODEL_DIR = 'model/'
    
    # ---------- NOTE: transformer neural network ----------
    HIDDEN_DIM = 32
    DROPOUT = 0.1
    NUM_LAYER = 3
    NUM_HEAD = 1
    ACTIVATION_FUNC = "relu"
    if ENV_TYPE == 'PA':  # * PA
        ACTION_TANH = False
    elif ENV_TYPE == 'MS':  # * MS
        ACTION_TANH = False


def get_config_dict():
    config = dict(vars(Config))
    config.pop('__doc__', None)
    config.pop('__weakref__', None)
    config.pop('__dict__', None)
    config.pop('__module__', None)
    return config


if __name__ == '__main__':
    config_ = get_config_dict()
    print(config_)