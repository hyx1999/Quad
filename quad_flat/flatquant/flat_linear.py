import torch
import torch.nn as nn
import torch.nn.functional as F

from flatquant.quant_utils import WeightQuantizer, ActivationQuantizer
from flatquant.flat_utils import kronecker_matmul

class FlatQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear):
        super(FlatQuantizedLinear, self).__init__()
        self.args = args
        self.linear = linear
        self.pod_size = linear.pod_rank

        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
        self.act_quantizer = ActivationQuantizer(bits=args.a_bits, sym=not(args.a_asym), lac=args.lac, groupsize=args.a_groupsize, )
        self.adapter = None

        self.lwc = args.lwc
        if self.lwc:
            lwc_dim = self.linear.weight.shape[0] if self.lwc else -1
            init_value = 4.
            self.clip_factor_w_max = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.clip_factor_w_min = nn.Parameter(torch.ones((lwc_dim, 1))*init_value, requires_grad=True)
            self.sigmoid = nn.Sigmoid()

        self._eval_mode = False

    @torch.no_grad()
    def add_adapter(self):
        self.adapter = nn.Linear(
            in_features=self.pod_size,
            out_features=self.linear.out_features,
            bias=False,
            dtype=self.linear.weight.dtype,
            device=self.linear.weight.device,
        )
        self.adapter.weight.data.zero_()

    @torch.no_grad()
    def merge_adapter(self):
        self.linear.weight.data[:, :self.pod_size] += self.adapter.weight.data
        self.adapter = None

    def apply_wclip(self, weight):
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax *= self.sigmoid(self.clip_factor_w_max)
        wmin *= self.sigmoid(self.clip_factor_w_min)
        weight = torch.clamp(weight, min=wmin, max=wmax)
        return weight

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            weight = kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        else:
            weight = qa_trans(weight, inv_t=True)
        return weight

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def _train_forward(self, hidden_states, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        weight_full, weight = weight[:, :self.pod_size], weight[:, self.pod_size:]
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        # learnable weight clipping 
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.pod_size > 0:
            weight_full = out_trans(weight_full.T).T
        
        # quantize weight
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)
        weight = torch.cat((weight_full, weight), dim=1)
        # quantize activation            
        hidden_states = torch.cat(
            (hidden_states[..., :self.pod_size], self.act_quantizer(hidden_states[..., self.pod_size:])), dim=-1)

        if out_trans is not None and self.linear.bias is not None:
            bias = out_trans(self.linear.bias.data)
        else:
            bias = self.linear.bias
        output = F.linear(hidden_states, weight, bias)
        return output

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        if not self._eval_mode:
            return self._train_forward(hidden_states, qa_trans=qa_trans, out_trans=out_trans)
        else:
            return self._eval_forward(hidden_states)

    def _eval_forward(self, hidden_states):
        x_dtype = hidden_states.dtype
        # hidden_states = self.act_quantizer(hidden_states).to(x_dtype)
        hidden_states = torch.cat(
            (hidden_states[..., :self.pod_size], self.act_quantizer(hidden_states[..., self.pod_size:]).to(x_dtype)), dim=-1)
        output = self.linear(hidden_states)
        if self.adapter is not None:
            output = output + self.adapter(hidden_states[..., :self.pod_size])
        return output

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        weight_full, weight = weight[:, :self.pod_size], weight[:, self.pod_size:]
        # quantization-adaptive transform
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        if out_trans is not None and self.pod_size > 0:
            weight_full = out_trans(weight_full.T).T
        if out_trans is not None and self.linear.bias is not None:
            self.linear.bias.data = out_trans(self.linear.bias.data)
        weight = torch.cat((weight_full, weight), dim=1)       
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True
