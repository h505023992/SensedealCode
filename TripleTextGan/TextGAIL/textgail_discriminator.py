import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchfly.text.rl import TextRLRewardFunc

import logging

class TextGAILDiscriminator(TextRLRewardFunc):
    def __init__(self, config, tokenizer, discriminator):
        self.config = config
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = torch.LongTensor([self.tokenizer.sep_token_id])


    def get_reward(self, states, actions,gen_reward=True):#states就是batch，action就是result
        device = next(self.discriminator.parameters()).device
        states = states.to(device)
        if gen_reward:
            actions = pad_sequence(actions, batch_first=True, padding_value=self.pad_token_id).to(device)
        else:
            actions = actions.to(device)
            #can = torch.cat((states, actions), dim=-1).to(device)
        reward = self.discriminator.get_reward(actions,states)
        return reward.cpu().numpy() #detach很重要，但是为啥要放在cpu上，疑惑

    def get_loss(self, states, actions,experts,weights=None):

        device = next(self.discriminator.parameters()).device
#        states = torch.stack(states)
        experts = pad_sequence([item for item in experts], batch_first=True, padding_value=self.pad_token_id)
        #ref = torch.cat((states, experts), dim=-1).to(device)
        actions = pad_sequence(actions, batch_first=True, padding_value=self.pad_token_id)
        #can = torch.cat((states, actions), dim=-1).to(device)
        if weights is not None:
            weights = weights.to(device)
        batch = {"reference":experts.to(device),"candidate": actions.to(device),"label":states.to(device)}
        results = self.discriminator(batch,weights)
        return results # reward loss