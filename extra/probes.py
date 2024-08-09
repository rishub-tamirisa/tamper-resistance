import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM
from dataclasses import dataclass
from accelerate.optimizer import move_to_device, AcceleratedOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing import Union
import torch.nn as nn
import math
import torch.nn.init as init


@dataclass
class CausalLMOutputWithPastWithProbes(CausalLMOutputWithPast):
    probe_outputs: torch.Tensor = None


# Simply inheriting AutoModelForCausalLM doesn't include layers from child classes when calling .from_pretrained()
class LlamaForCausalLMWithProbes(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_probes = self.config.num_hidden_layers + 1
        self.layer_ids = list(range(self.num_probes))
        self.probes: torch.nn.ModuleList = self.init_probes()
        self.probe_substr = "probe"
        self.unembed_name = "lm_head"
        self.embed_name = "embed_tokens"

    def init_probes(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        output = super().forward(*args, **kwargs)
        hidden_states = output.hidden_states
        probe_outputs = []

        for idx, layer_id in enumerate(self.layer_ids):
            probe_outputs += [self.probes[idx](hidden_states[layer_id])]

        return CausalLMOutputWithPastWithProbes(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            probe_outputs=probe_outputs,
        )

    def _is_probe(self, layer_name: str):
        return self.probe_substr in layer_name

    def _main_parameters(self):
        return [
            param
            for name, param in self.named_parameters()
            if not self._is_probe(name) and not self._is_frozen(name)
        ]

    def _probe_parameters(self):
        return [
            param for name, param in self.named_parameters() if self._is_probe(name)
        ]

    def init_optimizers(self, lr: float = 2e-5):
        self._main_optimizer = torch.optim.AdamW(
            self._main_parameters(),
            lr=lr,
        )
        self._probe_optimizer = torch.optim.AdamW(
            self._probe_parameters(),
            lr=lr,
        )

    def set_probe_requires_grad(self, setting: bool = True):
        for name, param in self.named_parameters():
            if self.probe_substr in name:
                param.requires_grad = setting

    def set_non_probe_requires_grad(self, setting: bool = True):
        for name, param in self.named_parameters():
            if self.probe_substr not in name and self.unembed_name not in name:
                param.requires_grad = setting

    def main_optimizer(self):
        return self._main_optimizer

    def probe_optimizer(self):
        return self._probe_optimizer

class LlamaForCausalLMWithWassersteinCriticProbes(LlamaForCausalLMWithProbes):
    def init_probes(self):
        self.probe_op = None
        return torch.nn.ModuleList(
            [
                MLPProbe(
                    res_stream_dim=self.config.hidden_size,
                    output_size=1,
                    res_stream_op=self.probe_op,
                )
                for _ in self.layer_ids
            ]
        )

class LlamaForCausalLMWithLinearProbes(LlamaForCausalLMWithProbes):
    def init_probes(self):
        self.num_probes = self.config.num_hidden_layers
        self.layer_ids = [1, 6, 11, 16, 21, 26, 31]
        self.probe_op = None  # "last"
        return torch.nn.ModuleList(
            [
                LinearProbe(
                    res_stream_dim=self.config.hidden_size,
                    output_size=self.config.vocab_size,
                    res_stream_op=self.probe_op,
                )
                for i in range(self.num_probes)
                if i in self.layer_ids
            ]
        )


class LinearProbe(torch.nn.Module):
    def __init__(self, res_stream_dim=4096, output_size=1, res_stream_op=None):
        super(LinearProbe, self).__init__()
        self.w = torch.nn.Parameter(torch.empty(output_size, res_stream_dim))
        self.b = torch.nn.Parameter(torch.zeros(output_size))

        self.norm = torch.nn.LayerNorm(res_stream_dim)
        self.res_stream_op = res_stream_op
        assert self.res_stream_op in [None, "mean", "last"]

    def forward(self, res_stream: torch.Tensor):
        res_stream = self.norm(res_stream)
        if self.res_stream_op == "mean":
            res_stream = res_stream.mean(dim=-2)
        if self.res_stream_op == "last":
            res_stream = res_stream[:, -1, :]
        return torch.nn.functional.linear(
            res_stream, F.normalize(self.w.data, dim=0), self.b
        )


class MLPProbe(torch.nn.Module):
    def __init__(self, res_stream_dim=4096, output_size=1, res_stream_op=None):
        super(MLPProbe, self).__init__()
        self.linear1 = torch.nn.Linear(res_stream_dim, 1024)
        self.linear2 = torch.nn.Linear(1024, 256)
        self.linear3 = torch.nn.Linear(256, output_size)
        self.res_stream_op = res_stream_op
        assert self.res_stream_op in [None, "mean", "last"]

        self.norm0 = torch.nn.LayerNorm(res_stream_dim)
        self.norm1 = torch.nn.LayerNorm(1024)
        self.norm2 = torch.nn.LayerNorm(256)

    def forward(self, res_stream: torch.Tensor):
        res_stream = self.norm0(res_stream)
        if self.res_stream_op == "mean":
            res_stream = res_stream.mean(dim=-2)
        if self.res_stream_op == "last":
            res_stream = res_stream[:, -1, :]

        res_stream = F.leaky_relu(
            self.norm1(
                F.linear(
                    res_stream,
                    torch.clamp(self.linear1.weight, -0.10, 0.10),
                    self.linear1.bias,
                )
            )
        )
        res_stream = F.leaky_relu(
            self.norm2(
                F.linear(
                    res_stream,
                    torch.clamp(self.linear2.weight, -0.10, 0.10),
                    self.linear2.bias,
                )
            )
        )
        return F.linear(
            res_stream, torch.clamp(self.linear3.weight, -0.10, 0.10), self.linear3.bias
        )
