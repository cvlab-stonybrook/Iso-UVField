import torch
import numpy as np
import torch.autograd.profiler as profiler
import pdb

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        # pdb.set_trace()
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, dim_in):
    embed_kwargs = {
        'include_input': True,
        'input_dims': dim_in,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim




class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    https://raw.githubusercontent.com/sxyu/pixel-nerf/master/src/model/code.py
    This is a faster version with negligible numerical difference 
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        # with profiler.record_function("positional_enc"):
        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], self.d_out-self.d_in)

        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)

        return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )
        
        
'''  
pe=PositionalEncoding(freq_factor=1.0)  
if __name__=="__main__":
    import pdb
    points=torch.rand(10000,3)
    embed_fn, input_ch = get_embedder(6, 3)
    err=0
    for i in range(10000):
        embpoints = embed_fn(points)
        embpoints_pn=pe(points)
        err+=torch.abs(embpoints-embpoints_pn).sum()
    print(err/10000)
''' 