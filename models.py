import transformers
from torch import nn, optim
import torch.nn.functional as F
import torch
from utils import off_diagonal, FullGatherLayer
transformers.logging.set_verbosity_error()

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.num_features = int(self.args.mlp.split("-")[-1])
        
        self.backbone = transformers.AutoModel.from_pretrained(args.arch)
        self.backbone.train()
        self.embedding = self.backbone.config.hidden_size
        self.projector = Projector(args, self.embedding)
        
        if self.args.use_param_weights:
            self.param = nn.ParameterDict({'sim_coeff':nn.Parameter(torch.rand(1) + self.args.sim_coeff),
                                           'std_coeff':nn.Parameter(torch.rand(1) + self.args.std_coeff),
                                           'cov_coeff':nn.Parameter(torch.rand(1) + self.args.cov_coeff),
                                           })

    def forward(self, batch, only_inference = False):
        
        if only_inference:    
            return self.projector(self.backbone(**batch).pooler_output)
        else:
            x = self.projector(self.backbone(**batch).pooler_output)
            y = self.projector(self.backbone(**batch).pooler_output)

            repr_loss = F.mse_loss(x, y)

            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

            cov_x = (x.T @ x) / (self.args.batch_size - 1)
            cov_y = (y.T @ y) / (self.args.batch_size - 1)
            cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

            if self.args.use_param_weights:
                loss = (
                self.param.sim_coeff * repr_loss
                + self.param.std_coeff * std_loss
                + self.param.cov_coeff * cov_loss                
                )
            else:
                loss = (
                self.args.sim_coeff * repr_loss
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
                )
            return {'loss':loss,
                    'repr_loss':repr_loss,
                    'std_loss':std_loss,
                    'cov_loss':cov_loss}

class BarlowModule(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased',
                                                           hidden_dropout_prob=args.dropout_prob,
                                                           attention_probs_dropout_prob=args.dropout_prob)
        self.bert.train()
        
        sizes = [self.bert.config.hidden_size] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(MLP(input_dim=sizes[i],hidden_size=sizes[i+1],output_dim=None))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        
    def forward(self,x):
        
        bert_output = self.bert(**x)
        embedding = bert_output.pooler_output
        projection = self.projector(embedding)
        
        return self.bn(projection)
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True) if output_dim is not None else nn.Identity(),
        )

    def forward(self, x):
        x = self.model(x)
        return x    
    
def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])