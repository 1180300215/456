# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import torch
import torch.nn as nn
import transformers

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_1.trajectory_gpt2 import GPT2Model


class GPTDecoder(nn.Module):    # 针对的是受控代理

    def __init__(
            self,
            conf,
            obs_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            **kwargs
    ):
        super().__init__()
        self.conf = conf
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        # NOTE: hyper-parameters for GPT-2
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # NOTE: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer_decoder = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.obs_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict obs or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.obs_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)


    def forward(self, obs, actions, rewards, returns_to_go, timesteps, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        batch_size, seq_length = obs.shape[0], obs.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(obs)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since obs predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer_decoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), obs (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state
        
        return state_preds, action_preds, return_preds

    def get_action(self, obs, actions, rewards, returns_to_go, timesteps,
                encoder_hidden_states=None, encoder_attention_mask=None):
        # we don't care about the past rewards in this model

        obs = obs.reshape(1, -1, self.obs_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            obs = obs[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-obs.shape[1]), torch.ones(obs.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=obs.device).reshape(1, -1)
            obs = torch.cat(
                [torch.zeros((obs.shape[0], self.max_length-obs.shape[1], self.obs_dim), device=obs.device), obs],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            obs, actions, None, returns_to_go, timesteps, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        
        return action_preds[0, -1]
    
    def load_model(self, param_path, device="cpu"):
        self.load_state_dict(
            torch.load(param_path, map_location=torch.device(device))
        )