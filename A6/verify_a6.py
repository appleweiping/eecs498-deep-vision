"""
Verification harness for Assignment 6 (VAE + GAN + network visualization +
style transfer).

Runs:
  1. shape/loss sanity checks for VAE, CVAE, GAN, DCGAN, style-transfer losses
  2. a REAL (modest) VAE training run on MNIST and saves generated samples
  3. a REAL (modest) vanilla-GAN training run on MNIST and saves samples

Figures are written to ../results/. CPU-only, kept small but real.
"""
import os
import sys
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
torch.manual_seed(0)
torch.set_num_threads(3)

import vae as vae_mod
import gan as gan_mod
import style_transfer as st

RESULTS = {}
FAILS = []
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES_DIR, exist_ok=True)


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")
    if not cond:
        FAILS.append(name)


# ---------------- Shape / loss sanity ----------------
print("=== VAE / GAN shape & loss checks ===")
x = torch.rand(4, 1, 28, 28)
model = vae_mod.VAE(input_size=784, latent_size=15)
x_hat, mu, logvar = model(x)
check("VAE forward shapes", x_hat.shape == x.shape and mu.shape == (4, 15))
loss = vae_mod.loss_function(x_hat, x, mu, logvar)
check("VAE loss scalar & finite", loss.dim() == 0 and torch.isfinite(loss))

c = F.one_hot(torch.randint(10, (4,)), 10).float()
cmodel = vae_mod.CVAE(input_size=784, num_classes=10, latent_size=15)
xc_hat, cmu, clogvar = cmodel(x, c)
check("CVAE forward shapes", xc_hat.shape == x.shape and cmu.shape == (4, 15))

noise = gan_mod.sample_noise(8, 96)
check("sample_noise range/shape",
      noise.shape == (8, 96) and noise.min() >= -1 and noise.max() <= 1)

D = gan_mod.discriminator()
G = gan_mod.generator()
fake = G(noise)
check("generator output shape", fake.shape == (8, 784))
logits = D(fake)
check("discriminator output shape", logits.shape == (8, 1))

lr = torch.randn(8, 1)
lf = torch.randn(8, 1)
dloss = gan_mod.discriminator_loss(lr, lf)
gloss = gan_mod.generator_loss(lf)
check("GAN losses finite", torch.isfinite(dloss) and torch.isfinite(gloss))
lsd = gan_mod.ls_discriminator_loss(lr, lf)
lsg = gan_mod.ls_generator_loss(lf)
check("LS-GAN losses finite", torch.isfinite(lsd) and torch.isfinite(lsg))

dcD = gan_mod.build_dc_classifier()
dcG = gan_mod.build_dc_generator()
dcfake = dcG(noise)
check("DCGAN generator shape", dcfake.shape == (8, 784))
check("DCGAN discriminator shape", dcD(dcfake).shape == (8, 1))

# ---------------- Style-transfer function checks ----------------
print("\n=== style transfer checks ===")
feat = torch.randn(1, 3, 4, 4)
g = st.gram_matrix(feat, normalize=False)
# manual reference
fl = feat.reshape(1, 3, 16)
gref = torch.bmm(fl, fl.transpose(1, 2))
check("gram_matrix", torch.allclose(g, gref, atol=1e-5))
cl = st.content_loss(2.0, feat, feat)
check("content_loss zero on identical", abs(cl.item()) < 1e-6)
tv = st.tv_loss(torch.randn(1, 3, 8, 8), 1.0)
check("tv_loss non-negative", tv.item() >= 0)

# ---------------- REAL VAE training on MNIST ----------------
print("\n=== REAL VAE training on MNIST (modest) ===")
transform = T.ToTensor()
train_set = torchvision.datasets.MNIST(
    "mnist_data/", train=True, download=True, transform=transform)
loader = DataLoader(train_set, batch_size=128, shuffle=True)

vae = vae_mod.VAE(input_size=784, latent_size=20)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
vae.train()
NUM_STEPS = 600
step = 0
last_losses = []
for epoch in range(3):
    for xb, _ in loader:
        opt.zero_grad()
        xh, mu, lv = vae(xb)
        loss = vae_mod.loss_function(xh, xb, mu, lv)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}: VAE loss (neg-ELBO/img) = {loss.item():.2f}")
        last_losses.append(loss.item())
        step += 1
        if step >= NUM_STEPS:
            break
    if step >= NUM_STEPS:
        break

final_vae_loss = sum(last_losses[-50:]) / 50
RESULTS["vae_final_negELBO_per_img"] = round(final_vae_loss, 3)
check("VAE loss decreased below 160", final_vae_loss < 160,
      f"(final={final_vae_loss:.1f})")

# sample generations from the prior
vae.eval()
with torch.no_grad():
    z = torch.randn(64, 20)
    samples = vae.decoder(z).reshape(64, 1, 28, 28)
grid = torchvision.utils.make_grid(samples, nrow=8)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title(f"VAE samples (neg-ELBO/img={final_vae_loss:.1f})")
plt.savefig(os.path.join(RES_DIR, "a6_vae_samples.png"), bbox_inches="tight", dpi=110)
plt.close()
print("  saved a6_vae_samples.png")

# reconstruction figure
with torch.no_grad():
    xb, _ = next(iter(loader))
    xb = xb[:16]
    xh, _, _ = vae(xb)
compare = torch.cat([xb, xh], dim=0)
grid = torchvision.utils.make_grid(compare, nrow=16)
plt.figure(figsize=(10, 2))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("VAE: top=input, bottom=reconstruction")
plt.savefig(os.path.join(RES_DIR, "a6_vae_recon.png"), bbox_inches="tight", dpi=110)
plt.close()
print("  saved a6_vae_recon.png")

# ---------------- REAL vanilla GAN training on MNIST ----------------
print("\n=== REAL vanilla GAN training on MNIST (modest) ===")
gan_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)
Dnet = gan_mod.discriminator()
Gnet = gan_mod.generator()
D_opt = gan_mod.get_optimizer(Dnet)
G_opt = gan_mod.get_optimizer(Gnet)
NOISE = 96
gstep = 0
GAN_STEPS = 800
for epoch in range(5):
    for xb, _ in gan_loader:
        # scale images to [-1, 1] to match generator's tanh output
        real = (xb.reshape(xb.size(0), -1) * 2 - 1)
        # --- train D ---
        D_opt.zero_grad()
        logits_real = Dnet(real)
        z = gan_mod.sample_noise(real.size(0), NOISE)
        fake = Gnet(z).detach()
        logits_fake = Dnet(fake)
        d_loss = gan_mod.discriminator_loss(logits_real, logits_fake)
        d_loss.backward()
        D_opt.step()
        # --- train G ---
        G_opt.zero_grad()
        z = gan_mod.sample_noise(real.size(0), NOISE)
        fake = Gnet(z)
        g_loss = gan_mod.generator_loss(Dnet(fake))
        g_loss.backward()
        G_opt.step()
        if gstep % 100 == 0:
            print(f"  step {gstep}: D={d_loss.item():.3f} G={g_loss.item():.3f}")
        gstep += 1
        if gstep >= GAN_STEPS:
            break
    if gstep >= GAN_STEPS:
        break

RESULTS["gan_final_D_loss"] = round(d_loss.item(), 3)
RESULTS["gan_final_G_loss"] = round(g_loss.item(), 3)
check("GAN losses finite after training",
      torch.isfinite(d_loss) and torch.isfinite(g_loss))

with torch.no_grad():
    z = gan_mod.sample_noise(64, NOISE)
    gen = Gnet(z).reshape(64, 1, 28, 28)
    gen = (gen + 1) / 2  # back to [0,1] for display
grid = torchvision.utils.make_grid(gen, nrow=8)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Vanilla GAN samples (MNIST)")
plt.savefig(os.path.join(RES_DIR, "a6_gan_samples.png"), bbox_inches="tight", dpi=110)
plt.close()
print("  saved a6_gan_samples.png")

# ---------------- save results ----------------
out_path = os.path.join(RES_DIR, "a6_generative.txt")
with open(out_path, "w") as f:
    f.write("EECS 498-007 A6 (VAE / GAN) verification\n" + "=" * 40 + "\n")
    f.write(f"failed checks: {FAILS}\n\n")
    f.write(json.dumps(RESULTS, indent=2))
print(f"\nSaved {out_path}")
print("=" * 50)
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL A6 CHECKS PASSED")
