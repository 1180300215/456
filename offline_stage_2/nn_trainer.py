# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import time
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from offline_stage_2.utils import LOG


class ResponsePolicyTrainer:

    def __init__(self,
                encoder,
                decoder,
                batch_size,
                encoder_optimizer,
                decoder_optimizer,
                encoder_scheduler,
                decoder_scheduler,
                get_batch_fn,
                loss_fn,
                config,):
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler
        
        self.get_batch_fn = get_batch_fn
        self.loss_fn = loss_fn
        
        self.config = config
        self.clip_grad = self.config["CLIP_GRAD"]
        self.diagnostics = dict()

        self.start_time = time.time()

    def train(self, num_update):
        train_losses = []
        logs = dict()

        train_start = time.time()
        self.encoder.train()
        self.decoder.train()
        for _ in range(num_update):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs

    def train_step(self):
        batch = self.get_batch_fn()
        n_o_e, a_e, r_e, timesteps_e, mask_e, o_d, a_d, r_d, rtg_d, timesteps_d, mask_d, o_e_d, a_e_d = batch
        a_d_target = a_d.detach().clone()
        

        token_embeds, token_mask = self.encoder.get_tokens(
            obs=n_o_e, 
            action=a_e, 
            reward=r_e, 
            timestep=timesteps_e, 
            attention_mask=mask_e,
        )

        _, a_preds, _ = self.decoder.forward(
            o_d, a_d, r_d, rtg_d[:,:-1], timesteps_d, attention_mask=mask_d,
            encoder_hidden_states=token_embeds, encoder_attention_mask=token_mask,
        )

        act_dim = a_preds.shape[2]
        a_preds = a_preds.reshape(-1, act_dim)[mask_d.reshape(-1) > 0]
        a_d_target = a_d_target.reshape(-1, act_dim)[mask_d.reshape(-1) > 0]

        loss = self.loss_fn(a_preds, a_d_target)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_grad)
        self.encoder_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_grad)
        self.decoder_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().clone().cpu().item()

        return loss.detach().cpu().item()

    def eval(self, eval_episodes, oppo_policy_list, env_and_oppo, eval_type='seen'):
        LOG.info(f'Evaluate against opponent policies: {oppo_policy_list}')
        logs = dict()
        LOG.info('Start evaluating...')
        self.encoder.eval()
        self.decoder.eval()

        eval_start = time.time()
        return_mean_list = []
        for oppo_id, oppo_name in enumerate(oppo_policy_list):
            self.eval_fn = eval_episodes(env_and_oppo["env"], env_and_oppo["oppo_policy"][oppo_id],
                                            self.config, oppo_id, oppo_name)
            outputs, return_mean = self.eval_fn(self.encoder, self.decoder)
            return_mean_list.append(return_mean)
            for k, v in outputs.items():
                logs[f'{eval_type}-evaluation/{k}'] = v

        logs[f'{eval_type}-evaluation/average_return_mean'] = np.mean(return_mean_list)
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        return logs

    def save_model(self, postfix, save_dir):
        encoder_model_name = '/res_encoder' + postfix
        torch.save(self.encoder.state_dict(),save_dir+encoder_model_name)  # model save
        LOG.info(f'RES-Encoder saved to {save_dir+encoder_model_name}')
        decoder_model_name = '/res_decoder' + postfix
        torch.save(self.decoder.state_dict(),save_dir+decoder_model_name)  # model save
        LOG.info(f'RES-Decoder saved to {save_dir+decoder_model_name}')
