from typing import Optional
import logging

from e3nn import o3

from data import AtomicDataDict, AtomicDataset
from nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from . import builder_utils

class EnergyNet(GraphModuleMixin, torch.nn.Module):
    def __init__(self,
                 num_layer=3,
                 ) 
        super().__init__()
        self.one_hot=OneHotAtomEncoding()
        self.spharm_edges=SphericalHarmonicEdgeAttrs()
        self.radial_basis=RadialBasisEdgeEncoding()    
        
        self.chemical_embedding=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        self.layer1_convnet=ConvNetLayer()
        self.layer2_convnet=ConvNetLayer()
        self.layer3_convnet=ConvNetLayer()
        
        self.q=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        self.k=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        self.v=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        
        self.attention=Attention()
        
        self.fc1=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        self.fc2=AtomwiseLinear(irreps_in={'node_features':'5x0e'},irreps_out='32x0e')
        
    def forward(x):
        
        #initial embedding
        s_nuclear=self.one_hot(x)
        
        x=self.layer3_convnet(data)
        
        
        
        q=self.q(x)
        k=self.k(x)
        v=self.v(x)
        x=self.attention(q,k,v)
        return x
    
    
class Attention(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.linear_q=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.linear_k=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.linear_v=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        
        self.attention=Attention()
    def forward(self, data: AtomicDataDict.Type, eps: float=1e-8) -> AtomicDataDict.Type:
        
        q=self.linear_q(data[self.field])
        k=self.linear_k(data[self.field])
        v=self.linear_v(data[self.field])
        d=q.shape[-1]
        dot=q@k.T
        A=torch.exp((dot-torch.max(dot))/d**0.5)
        norm=torch.sum(A,dim=-1.keepdim=True)+eps
        data[self.field]=(A/norm)@v
        
        return data

        
        
        