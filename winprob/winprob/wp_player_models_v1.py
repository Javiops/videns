"""Fixed small set encoders for the joint-player WP development study."""
import math
import torch
from torch import nn

ARMS=('global','deepset','attention','attention_position_shuffled')
SEEDS=(20260929,20260930,20261001)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.width=64;self.heads=4
        self.query=nn.Linear(64,64);self.key=nn.Linear(64,64);self.value=nn.Linear(64,64);self.output=nn.Linear(64,64)
        self.first_norm=nn.LayerNorm(64);self.second_norm=nn.LayerNorm(64)
        self.feedforward=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))

    def forward(self,query,context):
        batch,nq,_=query.shape;nk=context.shape[1]
        q=self.query(query).reshape(batch,nq,4,16).transpose(1,2)
        k=self.key(context).reshape(batch,nk,4,16).transpose(1,2)
        v=self.value(context).reshape(batch,nk,4,16).transpose(1,2)
        weights=torch.softmax(torch.matmul(q,k.transpose(-1,-2))/math.sqrt(16),dim=-1)
        combined=torch.matmul(weights,v).transpose(1,2).reshape(batch,nq,64)
        first=self.first_norm(query+self.output(combined))
        return self.second_norm(first+self.feedforward(first))


class PlayerWinModel(nn.Module):
    forecast_scope='retrospective_OMNI_observational_current_state'
    live_safe=False
    supports_counterfactual_actions=False

    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown fixed player-representation arm')
        self.arm=arm;self.check_input_contents=True
        if arm=='deepset':self.token_encoder=nn.Sequential(nn.Linear(35,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        elif arm.startswith('attention'):
            self.token_encoder=nn.Linear(35,64);self.blocks=nn.ModuleList([AttentionBlock(),AttentionBlock()])
            self.pool_query=nn.Parameter(torch.randn(1,1,64)*.02);self.pool=AttentionBlock()
        self.head=nn.Sequential(nn.Linear(237,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,global_features,players=None):
        if global_features.ndim!=2 or global_features.shape[1]!=173 or global_features.dtype!=torch.float32:
            raise ValueError('Finite float32 global173 matrix required')
        if self.check_input_contents and not bool(torch.isfinite(global_features).all()):raise ValueError('Nonfinite global inputs')
        if self.arm=='global':representation=global_features.new_zeros((len(global_features),64))
        else:
            if players is None or players.shape!=(len(global_features),10,35) or players.dtype!=torch.float32 or players.device!=global_features.device:
                raise ValueError('Aligned float32 ten-player35-feature tensor required')
            if self.check_input_contents and not bool(torch.isfinite(players).all()):raise ValueError('Nonfinite player inputs')
            tokens=self.token_encoder(players)
            if self.arm=='deepset':representation=tokens.mean(dim=1)
            else:
                for block in self.blocks:tokens=block(tokens,tokens)
                representation=self.pool(self.pool_query.expand(len(tokens),-1,-1),tokens)[:,0]
        return self.head(torch.cat((global_features,representation),dim=1)).squeeze(-1)


def parameter_count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
