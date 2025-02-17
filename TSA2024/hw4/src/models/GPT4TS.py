import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from src.models.DLbase import DLForecastModel
from torch.optim.adamw import AdamW
from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class GPT4TS(DLForecastModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = Model(args).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_len = configs.patch_len
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.gpt2 = GPT2Model.from_pretrained(
            "/home/nfs02/model/gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
        )  # loads a pretrained GPT-2 base model
        self.gpt2.h = self.gpt2.h[: configs.e_layers]

        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.cnt = 0

    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, "b l m -> b m l")

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = rearrange(x, "b m n p -> (b m) n p")

        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, "(b m) l -> b l m", b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
