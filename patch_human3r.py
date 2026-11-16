"""Build-time patch for the Human3R checkout (usage: python patch_human3r.py <Human3R dir>).

Human3R expects the compiled CUDA kernel curope (cuRoPE2D). We do not compile it (keeps the build GPU/nvcc free) and
fall back to the repo's pure-PyTorch RoPE2D, which breaks on the NEGATIVE token positions Human3R feeds the state
tokens (the CUDA kernel just evaluates cos/sin of the negative angle). This patch makes the fallback accept negative
positions by offsetting the lookup table; results are mathematically identical to the CUDA kernel.
"""
import sys

path = sys.argv[1].rstrip("/") + "/src/croco/models/pos_embed.py"
s = open(path).read()
rep = [
    ("""        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D, seq_len, device, dtype) not in self.cache:""",
     """        def get_cos_sin(self, D, seq_len, device, dtype, offset=0):
            if (D, seq_len, device, dtype, offset) not in self.cache:"""),
    ("""                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
""", """                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) + offset
"""),
    ("""                self.cache[D, seq_len, device, dtype] = (cos, sin)
            return self.cache[D, seq_len, device, dtype]""",
     """                self.cache[D, seq_len, device, dtype, offset] = (cos, sin)
            return self.cache[D, seq_len, device, dtype, offset]"""),
    ("""            cos, sin = self.get_cos_sin(
                D, int(positions.max()) + 1, tokens.device, tokens.dtype
            )""",
     """            pmin, pmax = int(positions.min()), int(positions.max())
            cos, sin = self.get_cos_sin(D, pmax - pmin + 1, tokens.device, tokens.dtype, pmin)
            positions = positions - pmin"""),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(path, "w").write(s)
print("patched", path)

# Second patch: utils/device.py to_gpu() copies small temp clones with non_blocking=True. A pending async copy from a
# freed pageable temp can race (observed: img_mask arrived all-False on one GPU run, identical run succeeded before).
dpath = sys.argv[1].rstrip("/") + "/src/dust3r/utils/device.py"
d = open(dpath).read()
a = "view[name] = [x.clone().to(device, non_blocking=True) for x in _view[name]]"
b = "view[name] = _view[name].clone().to(device, non_blocking=True)"
assert d.count(a) == 1 and d.count(b) == 1
d = d.replace(a, a.replace("non_blocking=True", "non_blocking=False")).replace(b, b.replace("non_blocking=True", "non_blocking=False"))
open(dpath, "w").write(d)
print("patched", dpath)
