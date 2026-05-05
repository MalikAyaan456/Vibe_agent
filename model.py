# ================================================================
#  AIR WAVES VIBE  ·  CELL 1  |  Build Full Model From Scratch
#  Paste this ENTIRE cell into Google Colab and run it.
# ================================================================

# ── Install ──────────────────────────────────────────────────────
!pip install einops av librosa soundfile accelerate -q

import os, math, json
import torch, numpy as np, librosa
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Tuple

# ── GPU Detection ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram:.0f} GB")
    SIZE = 'large' if vram >= 35 else 'medium' if vram >= 12 else 'small'
else:
    vram, SIZE = 0, 'small'
    print("No GPU — CPU mode (slow)")
print(f"Profile: {SIZE}\n")

# ── Config ────────────────────────────────────────────────────────
@dataclass
class Cfg:
    H      : int   = 128 if SIZE == 'large' else 64
    W      : int   = 128 if SIZE == 'large' else 64
    T      : int   = 32  if SIZE == 'large' else 16
    fps    : int   = 4
    sr     : int   = 22050
    dur    : float = 8.0
    n_mels : int   = 128
    hop    : int   = 256
    n_fft  : int   = 1024
    vae_ch : int   = 64  if SIZE != 'small' else 32
    vae_z  : int   = 8
    vae_mult: Tuple= (1, 2, 4)
    vae_nr : int   = 2
    u_ch   : int   = 256 if SIZE == 'large' else 128
    u_mult : Tuple = (1,2,4,4) if SIZE=='large' else (1,2,4)
    u_nr   : int   = 2
    n_heads: int   = 8
    a_ch   : int   = 128
    a_mult : Tuple = (1, 2, 4)
    T_diff : int   = 1000
    batch  : int   = 2 if SIZE == 'large' else 1
    accum  : int   = 4
    lr     : float = 1e-4
    warmup : int   = 500

C = Cfg()
lat_h = C.H // 4
lat_w = C.W // 4
print(f"Video  : {C.T}f × {C.H}×{C.W}  |  Latent: {C.vae_z}×{lat_h}×{lat_w}")
print(f"UNet   : base={C.u_ch}, mults={C.u_mult}, heads={C.n_heads}")
print(f"Audio  : {C.sr}Hz  n_mels={C.n_mels}\n")

# ── Primitives ────────────────────────────────────────────────────
def sinusoidal(t: torch.Tensor, dim: int, max_p: int = 10000):
    half = dim // 2
    freq = torch.exp(-math.log(max_p) * torch.arange(half, device=t.device) / half)
    ang  = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([ang.cos(), ang.sin()], dim=-1)

class Swish(nn.Module):
    def forward(self, x): return x * x.sigmoid()

def gnorm(ch): return nn.GroupNorm(min(32, ch), ch)
def conv(i, o, k=3, s=1, p=1): return nn.Conv2d(i, o, k, s, p)

# ── Residual Block ────────────────────────────────────────────────
class Res(nn.Module):
    def __init__(self, ic, oc, td=None):
        super().__init__()
        self.n1, self.c1 = gnorm(ic), conv(ic, oc)
        self.n2, self.c2 = gnorm(oc), conv(oc, oc)
        self.act  = Swish()
        self.skip = conv(ic, oc, 1, 1, 0) if ic != oc else nn.Identity()
        self.te   = nn.Sequential(Swish(), nn.Linear(td, oc*2)) if td else None
    def forward(self, x, t=None):
        h = self.c1(self.act(self.n1(x)))
        if t is not None and self.te:
            sc, sh = self.te(t).chunk(2, -1)
            h = h * (1 + sc[...,None,None]) + sh[...,None,None]
        return self.c2(self.act(self.n2(h))) + self.skip(x)

# ── Spatial Self-Attention ────────────────────────────────────────
class SAttn(nn.Module):
    def __init__(self, ch, h=8):
        super().__init__()
        self.h = h; self.sc = (ch // h) ** -0.5
        self.n = gnorm(ch)
        self.qkv = conv(ch, ch*3, 1, 1, 0)
        self.out  = conv(ch, ch, 1, 1, 0)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).chunk(3, 1)
        f = lambda t: rearrange(t, 'b (n d) h w -> b n (h w) d', n=self.h)
        a = (f(q) @ f(k).transpose(-2,-1)) * self.sc
        o = rearrange(a.softmax(-1) @ f(v), 'b n (h w) d -> b (n d) h w', h=H, w=W)
        return x + self.out(o)

# ── Temporal Self-Attention ───────────────────────────────────────
class TAttn(nn.Module):
    def __init__(self, ch, h=8):
        super().__init__()
        self.h = h; self.sc = (ch // h) ** -0.5
        self.n = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, ch*3)
        self.out  = nn.Linear(ch, ch)
    def forward(self, x):
        q, k, v = self.qkv(self.n(x)).chunk(3, -1)
        f = lambda t: rearrange(t, 'n t (h d) -> n h t d', h=self.h)
        o = rearrange((f(q) @ f(k).transpose(-2,-1) * self.sc).softmax(-1) @ f(v),
                      'n h t d -> n t (h d)')
        return x + self.out(o)

# ── Temporal Conv ─────────────────────────────────────────────────
class TConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c = nn.Sequential(gnorm(ch), Swish(), nn.Conv1d(ch, ch, 3, 1, 1))
        nn.init.zeros_(self.c[-1].weight); nn.init.zeros_(self.c[-1].bias)
    def forward(self, x, T):
        BT, C, H, W = x.shape; B = BT // T
        r = rearrange(x, '(b t) c h w -> (b h w) c t', b=B, t=T)
        return x + rearrange(self.c(r), '(b h w) c t -> (b t) c h w', b=B, h=H, w=W)

# ── Pseudo-3D Block ───────────────────────────────────────────────
class P3D(nn.Module):
    def __init__(self, ic, oc, td, heads, attn=True):
        super().__init__()
        self.r  = Res(ic, oc, td)
        self.tc = TConv(oc)
        self.sa = SAttn(oc, heads) if attn else nn.Identity()
        self.ta = TAttn(oc, heads) if attn else nn.Identity()
    def forward(self, x, te, T):
        x = self.r(x, te);  x = self.tc(x, T);  x = self.sa(x)
        BT, C, H, W = x.shape; B = BT // T
        r = rearrange(x, '(b t) c h w -> (b h w) t c', b=B, t=T)
        r = self.ta(r)
        return rearrange(r, '(b h w) t c -> (b t) c h w', b=B, h=H, w=W)

class Dn(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = conv(ch, ch, 3, 2, 1)
    def forward(self, x): return self.c(x)

class Up(nn.Module):
    def __init__(self, ch): super().__init__(); self.c = conv(ch, ch, 3, 1, 1)
    def forward(self, x): return self.c(F.interpolate(x, scale_factor=2, mode='nearest'))

# ── Video VAE ─────────────────────────────────────────────────────
class VideoVAE(nn.Module):
    def __init__(self, C):
        super().__init__()
        ch, z, ml, nr = C.vae_ch, C.vae_z, C.vae_mult, C.vae_nr
        self.enc_in = conv(3, ch)
        enc = []; cur = ch
        for i, m in enumerate(ml):
            oc = ch * m
            for _ in range(nr): enc.append(Res(cur, oc)); cur = oc
            if i < len(ml) - 1: enc.append(Dn(cur))
        self.enc     = nn.ModuleList(enc)
        self.enc_mid = nn.ModuleList([Res(cur,cur), SAttn(cur), Res(cur,cur)])
        self.enc_out = nn.Sequential(gnorm(cur), Swish(), conv(cur, z*2, 1,1,0))
        top = ch * ml[-1]
        self.dec_in  = conv(z, top, 1, 1, 0)
        self.dec_mid = nn.ModuleList([Res(top,top), SAttn(top), Res(top,top)])
        dec = []; cur = top
        for i, m in enumerate(reversed(ml)):
            oc = ch * m
            for _ in range(nr+1): dec.append(Res(cur, oc)); cur = oc
            if i < len(ml) - 1: dec.append(Up(cur))
        self.dec     = nn.ModuleList(dec)
        self.dec_out = nn.Sequential(gnorm(cur), Swish(), conv(cur, 3), nn.Tanh())
        self.kl_w    = 1e-6

    def encode(self, x):
        B, T = x.shape[0], x.shape[1]
        h = self.enc_in(rearrange(x, 'b t c h w -> (b t) c h w'))
        for l in self.enc:     h = l(h)
        for l in self.enc_mid: h = l(h)
        mu, lv = self.enc_out(h).chunk(2, 1)
        return mu, lv.clamp(-30, 20)

    def decode(self, z, T):
        h = self.dec_in(z)
        for l in self.dec_mid: h = l(h)
        for l in self.dec:     h = l(h)
        return rearrange(self.dec_out(h), '(b t) c h w -> b t c h w', t=T)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        recon = self.decode(z, x.shape[1])
        kl    = -0.5 * (1 + lv - mu**2 - lv.exp()).mean()
        return recon, kl, z

# ── Video Diffusion UNet ──────────────────────────────────────────
class VideoUNet(nn.Module):
    def __init__(self, C):
        super().__init__()
        ch, ml, nh, z = C.u_ch, C.u_mult, C.n_heads, C.vae_z
        td = ch * 4
        self.te = nn.Sequential(nn.Linear(ch, td), Swish(), nn.Linear(td, td))
        self.ci = conv(z, ch)
        chs = [ch * m for m in ml]
        io  = list(zip([ch] + chs[:-1], chs))
        self.downs = nn.ModuleList(); self.dsamp = nn.ModuleList()
        for i, (ic, oc) in enumerate(io):
            at = (i >= len(ml) - 2)
            self.downs.append(nn.ModuleList([P3D(ic,oc,td,nh,at), P3D(oc,oc,td,nh,at)]))
            self.dsamp.append(Dn(oc) if i < len(ml)-1 else nn.Identity())
        bot = chs[-1]
        self.mid = nn.ModuleList([P3D(bot,bot,td,nh,True), P3D(bot,bot,td,nh,True)])
        self.ups = nn.ModuleList(); self.usamp = nn.ModuleList()
        cur = bot
        for i, (ic, oc) in enumerate(reversed(io)):
            at = (i < 2)
            self.ups.append(nn.ModuleList([P3D(cur+oc,oc,td,nh,at),
                                           P3D(oc+oc, ic,td,nh,at)]))
            self.usamp.append(Up(ic) if i < len(ml)-1 else nn.Identity())
            cur = ic
        self.no = gnorm(ch); self.co = conv(ch, z, 1, 1, 0)
        nn.init.zeros_(self.co.weight); nn.init.zeros_(self.co.bias)

    def forward(self, x, t):
        B, T = x.shape[:2]
        x  = rearrange(x, 'b t c h w -> (b t) c h w')
        te = repeat(self.te(sinusoidal(t, self.te[0].in_features)),
                    'b d -> (b t) d', t=T)
        x = self.ci(x); sk = []
        for (b1,b2), ds in zip(self.downs, self.dsamp):
            x=b1(x,te,T); sk.append(x)
            x=b2(x,te,T); sk.append(x)
            x=ds(x)
        for m in self.mid: x = m(x, te, T)
        for (b1,b2), us in zip(self.ups, self.usamp):
            x = torch.cat([x, sk.pop()], 1); x = b1(x, te, T)
            x = torch.cat([x, sk.pop()], 1); x = b2(x, te, T)
            x = us(x)
        return rearrange(self.co(Swish()(self.no(x))),
                         '(b t) c h w -> b t c h w', b=B, t=T)

# ── Audio Residual Block ──────────────────────────────────────────
class AR(nn.Module):
    def __init__(self, ic, oc, td=None):
        super().__init__()
        self.n1=nn.GroupNorm(min(32,ic),ic); self.c1=nn.Conv2d(ic,oc,3,1,1)
        self.n2=nn.GroupNorm(min(32,oc),oc); self.c2=nn.Conv2d(oc,oc,3,1,1)
        self.act=Swish(); self.skip=nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity()
        self.te =nn.Sequential(Swish(), nn.Linear(td,oc)) if td else None
    def forward(self, x, t=None):
        h = self.c1(self.act(self.n1(x)))
        if t is not None and self.te: h = h + self.te(t)[...,None,None]
        return self.c2(self.act(self.n2(h))) + self.skip(x)

# ── Audio UNet  (FIX: match skip sizes exactly before cat) ────────
class AudioUNet(nn.Module):
    def __init__(self, C):
        super().__init__()
        ch, ml = C.a_ch, C.a_mult; td = ch * 4
        self.te = nn.Sequential(nn.Linear(ch,td), Swish(), nn.Linear(td,td))
        self.ci = nn.Conv2d(1, ch, 3, 1, 1)
        chs = [ch*m for m in ml]
        io  = list(zip([ch]+chs[:-1], chs))
        self.downs = nn.ModuleList(); self.dsamp = nn.ModuleList()
        for i,(ic,oc) in enumerate(io):
            self.downs.append(nn.ModuleList([AR(ic,oc,td), AR(oc,oc,td)]))
            self.dsamp.append(nn.Conv2d(oc,oc,3,2,1) if i<len(ml)-1 else nn.Identity())
        bot = chs[-1]
        self.mid = nn.ModuleList([AR(bot,bot,td), AR(bot,bot,td)])
        self.ups  = nn.ModuleList(); self.usamp = nn.ModuleList()
        cur = bot
        for i,(ic,oc) in enumerate(reversed(io)):
            self.ups.append(nn.ModuleList([AR(cur+oc,oc,td), AR(oc+oc,ic,td)]))
            self.usamp.append(nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),
                                            nn.Conv2d(ic,ic,3,1,1))
                              if i<len(ml)-1 else nn.Identity())
            cur = ic
        self.no = nn.GroupNorm(min(32,ch),ch); self.co = nn.Conv2d(ch,1,3,1,1)
        nn.init.zeros_(self.co.weight); nn.init.zeros_(self.co.bias)

    @staticmethod
    def _match(x, ref):
        """Interpolate x to match ref's spatial size if they differ (fixes odd dims)."""
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='nearest')
        return x

    def forward(self, x, t):
        te = self.te(sinusoidal(t, self.te[0].in_features))
        x  = self.ci(x); sk = []
        for (b1,b2), ds in zip(self.downs, self.dsamp):
            x=b1(x,te); sk.append(x)
            x=b2(x,te); sk.append(x)
            x=ds(x)
        for m in self.mid: x = m(x, te)
        for (b1,b2), us in zip(self.ups, self.usamp):
            s = sk.pop(); x = self._match(x, s); x = torch.cat([x, s], 1); x = b1(x, te)
            s = sk.pop(); x = self._match(x, s); x = torch.cat([x, s], 1); x = b2(x, te)
            x = us(x)
        return self.co(F.silu(self.no(x)))

# ── Cosine DDPM Scheduler ─────────────────────────────────────────
class DDPM(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        s   = torch.arange(T+1).float() / T
        ab  = torch.cos((s+.008)/1.008 * math.pi/2)**2
        ab  = ab / ab[0]          # FIX: out-of-place to avoid memory alias error
        b   = (1 - ab[1:]/ab[:-1]).clamp(0, .999)
        a   = 1 - b;  ac = a.cumprod(0)
        acp = F.pad(ac[:-1], (1,0), value=1.)
        for nm, v in [('b',b),('ac',ac),('sac',ac.sqrt()),
                      ('s1ac',(1-ac).sqrt()),('pv',b*(1-acp)/(1-ac))]:
            self.register_buffer(nm, v)
        self.T = T

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        s  = self.sac[t].view(-1,  *([1]*(x0.ndim-1)))
        s1 = self.s1ac[t].view(-1, *([1]*(x0.ndim-1)))
        return s*x0 + s1*noise, noise

    @torch.no_grad()
    def sample(self, model_fn, shape, steps=50, device='cuda'):
        x  = torch.randn(shape, device=device)
        ts = torch.linspace(self.T-1, 0, steps, dtype=torch.long)
        for i, tv in enumerate(ts):
            t   = tv.expand(shape[0]).to(device)
            eps = model_fn(x, t)
            ac  = self.ac[tv]
            acp = self.ac[ts[i+1]] if i < steps-1 else torch.ones(1, device=device)
            x0  = ((x - (1-ac).sqrt()*eps) / ac.sqrt()).clamp(-1, 1)
            x   = acp.sqrt()*x0 + (1-acp).sqrt()*eps
        return x

# ── Master Model: Air Waves Vibe ──────────────────────────────────
class AirWavesVibe(nn.Module):
    def __init__(self, C: Cfg):
        super().__init__()
        self.C       = C
        self.vae     = VideoVAE(C)
        self.vid_net = VideoUNet(C)
        self.aud_net = AudioUNet(C)
        self.sched   = DDPM(C.T_diff)

    def video_loss(self, vid):
        B, T = vid.shape[:2]
        recon, kl, z = self.vae(vid)
        recon_l = F.l1_loss(recon, vid)
        zs = rearrange(z, '(b t) c h w -> b t c h w', b=B, t=T)
        t  = torch.randint(0, self.sched.T, (B,), device=vid.device)
        zn, noise = self.sched.add_noise(zs, t)
        diff_l = F.mse_loss(self.vid_net(zn, t), noise)
        total  = recon_l + self.vae.kl_w * kl + diff_l
        return total, recon_l.item(), kl.item(), diff_l.item()

    def audio_loss(self, mel):
        B = mel.shape[0]
        t  = torch.randint(0, self.sched.T, (B,), device=mel.device)
        mn, noise = self.sched.add_noise(mel, t)
        return F.mse_loss(self.aud_net(mn, t), noise)

    @torch.no_grad()
    def generate(self, n=1, steps=50):
        dev = next(self.parameters()).device; C = self.C
        lat   = (n, C.T, C.vae_z, C.H//4, C.W//4)
        zg    = self.sched.sample(lambda x,t: self.vid_net(x,t), lat, steps, dev)
        zf    = rearrange(zg, 'b t c h w -> (b t) c h w')
        frames= self.vae.decode(zf, C.T)
        frames= ((frames.clamp(-1,1) + 1) / 2).cpu()
        at    = int(C.dur * C.sr / C.hop) + 1
        mel_g = self.sched.sample(lambda x,t: self.aud_net(x,t),
                                  (n, 1, C.n_mels, at), steps, dev)
        return frames, mel_g.cpu()

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

# ── Build & Test ──────────────────────────────────────────────────
print("Building Air Waves Vibe…")
model = AirWavesVibe(C).to(device)
print(f"  Total  : {model.n_params():.1f} M params")
print(f"  VAE    : {sum(p.numel() for p in model.vae.parameters())/1e6:.1f} M")
print(f"  VidUNet: {sum(p.numel() for p in model.vid_net.parameters())/1e6:.1f} M")
print(f"  AudUNet: {sum(p.numel() for p in model.aud_net.parameters())/1e6:.1f} M")

print("\nRunning forward-pass sanity check…")
with torch.no_grad():
    tv  = torch.randn(1, C.T, 3, C.H, C.W, device=device)
    tm  = torch.randn(1, 1, C.n_mels, int(C.dur*C.sr/C.hop)+1, device=device)
    tt  = torch.randint(0, C.T_diff, (1,), device=device)
    recon, kl, z = model.vae(tv)
    print(f"  VAE      : {list(tv.shape)} → z{list(z.shape)} → recon{list(recon.shape)}")
    zs = rearrange(z, '(b t) c h w -> b t c h w', b=1, t=C.T)
    pv = model.vid_net(zs, tt)
    print(f"  VideoUNet: {list(zs.shape)} → {list(pv.shape)}")
    pa = model.aud_net(tm, tt)
    print(f"  AudioUNet: {list(tm.shape)} → {list(pa.shape)}")
print("✔ All checks passed!\n")

# ── Save to Google Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
SAVE = "/content/drive/MyDrive/Air waves vibe"
os.makedirs(SAVE, exist_ok=True)
torch.save(model.state_dict(), f"{SAVE}/model_init.pt")
cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k,v in C.__dict__.items()}
with open(f"{SAVE}/config.json", "w") as f:
    json.dump(cfg_dict, f, indent=2)
print(f"✔ Saved to: {SAVE}")
print("─"*60)
print("  CELL 1 DONE — paste & run Cell 2 to start training")
print("─"*60)
