# data/modelo.py
import torch
import torch.nn as nn

class MembershipFunctionLayer(nn.Module):
    def __init__(self, n_inputs, n_mf):
        super(MembershipFunctionLayer, self).__init__()
        self.n_mf = n_mf
        self.centers = nn.Parameter(torch.rand(n_inputs, n_mf))
        self.widths = nn.Parameter(torch.rand(n_inputs, n_mf))

    def forward(self, x):
        x_expanded = x.unsqueeze(2)
        centers_expanded = self.centers.unsqueeze(0)
        widths_expanded = self.widths.unsqueeze(0)
        mf_out = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * widths_expanded ** 2))
        return mf_out.reshape(x.size(0), -1)

class RuleLayer(nn.Module):
    def __init__(self, n_rules):
        super(RuleLayer, self).__init__()
        self.weights = nn.Parameter(torch.rand(n_rules, 1))

    def forward(self, mf_out):
        product = torch.prod(mf_out, dim=1, keepdim=True)
        product = product.repeat(1, self.weights.size(0) // product.size(1))
        return torch.matmul(product, self.weights)

class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_mf):
        super(ANFIS, self).__init__()
        self.n_rules = n_mf ** n_inputs  # Total de reglas
        self.mf_layer = MembershipFunctionLayer(n_inputs, n_mf)
        self.rule_layer = RuleLayer(self.n_rules)

    def forward(self, x):
        mf_out = self.mf_layer(x)
        output = self.rule_layer(mf_out)
        return output
