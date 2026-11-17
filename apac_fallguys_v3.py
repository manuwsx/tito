# =============================================================================
# "Fall Guys" stochastic Mean Field Game solved with APAC-Net (Lin et al., PNAS 2021)
# Every design choice below is copied from the official repo (atlin23/apac-net) unless a
# comment says otherwise: networks, congestion estimator, Laplacian trick,
# losses, optimisers and learning rates. (Hamiltonian: see HAM below. Walls: see "Obstacles".)
#
# Courses are presets in COURSES (pick one with COURSE). Default "the_fork": a short island blocks the
# straight line so the crowd must split, and one narrow door before the finish makes it merge again.
#
# MFG system (horizon T = 1, minimisation convention, as in the paper):
#   HJB : -d_t phi - nu Lap phi + H(x, grad phi) = f(x, rho, t),   phi(x, T) = g(x)
#   FP  :  d_t rho - nu Lap rho - div(rho grad_p H(x, grad phi)) = 0,   rho(x, 0) = rho_0
# Walls, the pillar and the hammers are COSTS in f (very high inside): the networks learn to avoid them.
# Saddle point:  inf_rho sup_phi  E_{rho_0}[phi(.,0)] + int int (d_t phi + nu Lap phi - H) rho + int F(rho_t) dt
# =============================================================================
import os
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SEED = int(os.environ.get("SEED", "0"))   # every run used seed 0, so every run started from the SAME tilt
torch.manual_seed(SEED)
# DEVICE=cpu / DEVICE=cuda picks the device and THREADS the number of CPU threads (test.py uses both to train
# some runs on CPU cores next to the GPU: the networks are tiny, so the GPU is mostly waiting for Python).
device = torch.device(os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True     # TF32 matrix products: faster on recent GPUs, accurate enough
    torch.backends.cudnn.allow_tf32 = True

# ================================ race courses =================================
# A wall is a box (x0, x1, y0, y1), optionally with a 5th entry that multiplies the push out penalty (W_PUSH) for that box; a hammer is (x, y_mid, amp, freq, t0) moving as
# y(t) = y_mid + amp sin(2 pi freq (t - t0)). Use |y| = 9 for "extends off the map".
COURSES = {
    # The Fork: a SHORT island right in the straight line forces a choice, top or bottom. The lanes are
    # short and wide (0.6 < |y| < 2.0) so passing is clearly worth it, and everything is mirror
    # symmetric, so only the starting height and congestion decide who goes where. One narrow door
    # before the finish makes the two streams merge again.
    # speed 7.8: the route through the NEAREST lane (about 7.3 long) just fits in T = 1, while crossing
    # over to the far lane (about 1 longer) means missing the goal. With speed 9 there was so much spare
    # time that the detour was free, and the whole crowd took the top lane.
    "the_fork": dict(
        symmetric=True,
        start=(-3.2, 0.0), start_std=(0.25, 0.9), goal=(3.6, 0.0), speed=7.8, effort=0.25, draw_t=0.3, wd_g=0.0,
        boxes=[(-1.2, 0.2, -0.6, 0.6, 2.0),                       # central island (2x: agents split around it)
               (-1.2, 0.2, 2.0, 9.0), (-1.2, 0.2, -9.0, -2.0),     # outer lane walls
               (1.6, 1.9, 0.55, 9.0), (1.6, 1.9, -9.0, -0.55)],    # final door, gap |y| < 0.55
        hammers=[]),
    # Same fork plus one small hammer per lane, mirrored (antiphase). Each sweeps across its lane,
    # so at any moment one side of the lane is open: agents must time their run or hug a wall.
    # Faster hammers (period 0.67 instead of 1) that are thinner, so an opening comes round quickly,
    # plus the rush cost below: agents now dash through the first opening instead of waiting a full
    # cycle. The crowd reaches x = -0.5 at about t = 0.34, when both hammers swing OUTWARDS, leaving a
    # strip along the island. Mirror symmetry already forces the split, so the time budget can be
    # generous again (8 * 1.3 = 10.4): enough to clear the hammers AND the final door.
    "the_fork_hammers": dict(
        symmetric=True,
        start=(-3.2, 0.0), start_std=(0.25, 0.9), goal=(3.6, 0.0), speed=8.0, effort=0.35, draw_t=0.34, T=1.3, wd_g=0.0,
        boxes=[(-1.2, 0.2, -0.7, 0.7, 2.0), (-1.2, 0.2, 2.0, 9.0), (-1.2, 0.2, -9.0, -2.0),
               (1.6, 1.9, 0.55, 9.0), (1.6, 1.9, -9.0, -0.55)],
        # amplitude 0.35 (was 0.5): the rim now just reaches the island (|y| = 0.7) and the outer wall (|y| = 2.0).
        # With 0.5 the disk sank 0.15 INTO both walls, and agents caught there were squashed between hammer
        # and wall with nowhere to go (4% of the SDE agents "inside an obstacle" in the last run).
        hammers=[(-0.5, 1.35, 0.35, 1.5, 0.30), (-0.5, -1.35, -0.35, 1.5, 0.30)], hammer_r=0.3),
    # One wall, one gap of width d = 0.6: the classic bottleneck. Expect the crowd to compress into
    # an arc (a "clog") in front of the gap and leave it as a narrow jet.
    "one_gap": dict(
        symmetric=True,
        start=(-3.0, 0.0), start_std=(0.25, 0.9), goal=(3.0, 0.0), speed=8.0, effort=0.3, draw_t=0.5, wd_g=0.0,
        boxes=[(-0.15, 0.15, 0.3, 9.0), (-0.15, 0.15, -9.0, -0.3)],
        hammers=[]),
    # Same gap plus a small square pillar of side D = 0.4 placed a distance r = 0.5 in front of it
    # (your diagram). Known from crowd evacuation studies: such a pillar can split the arc in front
    # of the exit into two streams and reduce clogging. Does the mean field equilibrium agree?
    # (The 5th entry 2.0 doubles the push out penalty for generator mass inside the pillar.)
    "one_gap_pillar": dict(
        symmetric=True,
        start=(-3.0, 0.0), start_std=(0.25, 0.9), goal=(3.0, 0.0), speed=8.0, effort=0.3, draw_t=0.5, wd_g=0.0,
        boxes=[(-0.15, 0.15, 0.3, 9.0), (-0.15, 0.15, -9.0, -0.3),
               (-0.15 - 0.5 - 0.4, -0.15 - 0.5, -0.2, 0.2, 2.0)],   # pillar: x in [-1.05, -0.65], |y| < 0.2
        hammers=[]),
    # Previous course: two side lanes first, one hammer in the middle, central door last.
    "hammer_gates": dict(
        start=(-3.0, 0.0), start_std=(0.25, 0.9), goal=(3.5, 0.0), speed=8.0, effort=0.25, draw_t=0.4,
        boxes=[(-1.65, -1.35, 1.9, 9.0), (-1.65, -1.35, -9.0, -1.9), (-1.65, -1.35, -0.6, 0.6),
               (1.35, 1.65, 0.5, 9.0), (1.35, 1.65, -9.0, -0.5)],
        hammers=[(0.0, 0.0, 1.2, 1.0, 0.4)], hammer_r=0.55, wd_g=0.0),
}
COURSE = os.environ.get("COURSE", "the_fork")   # set the COURSE environment variable to pick another preset
C = COURSES[COURSE]

# DIM=50 (environment variable) runs the same game in 50 dimensions. As in the repo's high dimensional runs, the
# obstacles, the goal and every cost act only on the first two coordinates (x1, x2); the other coordinates
# start as a Gaussian around 0 and diffuse, and the networks must learn to ignore them. The plots show (x1, x2).
# Cost: the Laplacian trick of hjb_terms repeats the batch DIM times, so a step costs about DIM / 2 times more.
DIM = int(os.environ.get("DIM", "2"))
T, NU = C.get("T", 1.0), 0.05  # horizon (repo TT) and noise (repo default 0.1; 0.05 = less jitter of lone agents)
# Hamiltonian. "quadratic": H(p) = |p|^2 / (2 effort), i.e. running cost L(v) = effort |v|^2 / 2: moving costs
# effort, so a longer route costs more (the paper's quadcopter example uses this L). "speed_limit": the repo
# bottleneck H(p) = c|p|, i.e. any speed up to c is FREE: a detour costs nothing as long as you still arrive
# by T. That is why the crowd never cared which lane it took: going round the far side was free.
HAM = "soft_cap"   # "soft_cap" (default), "capped", "quadratic", "speed_limit" (repo): see ham()
# Why the cap: people have a top speed. It also keeps an SDE step (about VMAX * dt) shorter than half a
# wall, so even a wrong control cannot carry an agent across a wall in one step.
EFFORT = C.get("effort", 0.3)    # lambda in L(v) = lambda |v|^2 / 2; typical speed is about |grad phi| / lambda
VMAX = C.get("vmax", 1.5 * C["speed"])   # top speed for HAM = "capped" (about 12: 50% above cruising speed)
TAIL = 10.0                  # soft_cap: running faster than VMAX costs TAIL times more effort than normal
# Why "soft_cap" and not "capped": with a HARD top speed, H grows only linearly in |p|, and then the
# phi player's objective has no upper bound as soon as the generator moves any mass faster than VMAX
# (phi can raise its gradient without limit). That is exactly the runaway in the last fork runs
# (loss_phi falling in a straight line to -150, then a crash). A steep but finite cost above VMAX keeps
# H superlinear, so the saddle problem stays well posed, and sprinting through thin walls stays too
# expensive to pay off.
SPEED, EPS_H = C["speed"], 0.3   # speed_limit option only: c in H(p) = c|p|; EPS_H smooths |p| at 0
START, START_STD = C["start"], C["start_std"]
GOAL, FINISH_R = C["goal"], 0.75
GOAL_ZONE_R = 0.5            # inside this radius the goal cost becomes quadratic (Huber, see g): the pull fades out
                             # smoothly at the star, so agents settle instead of overshooting. (A flat zone made
                             # them stop at its edge, half outside the finish circle.)
W_GOAL = 2.0                 # repo: psi_scale (doubled: a stronger pull through the obstacles)
# ----- Obstacles: all of them are costs in f, which phi and G learn from (as in APAC-Net) -----
# Being INSIDE a wall box or a hammer costs W_IN per unit time (very high: crossing a 0.3 wide wall then
# costs about 25, three times the longest detour, so cutting through is never worth it). Close to a wall
# there is a mild cost W_SKIRT (people keep a little distance), close to a hammer a stronger one (W_HAMMER).
# Nothing is masked: the HJB with this f is trained everywhere, inside the obstacles too, and G's Lagrangian
# contains the same f at every sample.
# What went wrong before, with walls at W_WALL = 150 and a plain network (measured on the last runs):
#   1. the cost was low enough to PAY: inside a thin wall phi just sloped down towards the goal side
#      ("crossing costs about 8, carry on"), and the generator sent mass through;
#   2. the network could not turn that steep slope off at the wall face. It overshot into a dip just behind
#      the wall, where the control pointed back at the wall: agents coming out of a door were pulled back
#      (the rubber banding) and parked behind walls and door posts.
# Three things fix it, all in the spirit of the method:
#   a. phi gets one explicit term for the inside of an obstacle, a (learnable) slope times the depth, exactly
#      as APAC-Net builds the terminal cost g into phi (see Phi): the network no longer has to draw a 0.3 wide
#      ridge 12 high, it only learns what happens OUTSIDE obstacles, i.e. everything the agents actually do;
#   b. at a wall face phi may not send agents INTO the wall (LAMBDA_FACE). A network only sees the HJB at
#      sample points, and "everybody runs into the face from both sides" satisfies it at every sample point:
#      this condition is the piece of the HJB that sample points cannot see;
#   c. the population pays for mass inside an obstacle (W_OFF + W_PUSH * depth, so its gradient shows the
#      way out) and for a path that jumps through a wall (tunnel_cost): mass cannot pass through a wall.
XL, YL = (-4.0, 4.5), (-3.0, 3.0)   # the map (plots, exploration points, wall faces)
W_IN = C.get("w_in", 2000.0)  # cost per unit time inside a wall box or a hammer (enters f for phi AND G)
W_SKIRT, R_SKIRT = 30.0, 0.2 # cost NEAR a wall, W_SKIRT (1 - distance / R_SKIRT)^2 within R_SKIRT of it. Tested on
                             # the pillar course: with 10, 8% of the agents still ran into a wall; with 30, 1.4%
W_PUSH, W_OFF = 750.0, 1000.0   # population only: W_OFF + W_PUSH * depth for mass inside an obstacle (push_out)
LAMBDA_FACE, N_FACE = 5.0, 256   # wall face condition on phi: weight, and points on the wall faces per step
IN_WEIGHT = 50.0             # an HJB residual inside an obstacle counts IN_WEIGHT / (IN_WEIGHT + W_IN): there a
                             # small error in the slope is a huge error in H, and nobody is there anyway
NEAR_WALL, NEAR_R = 0.3, 0.3 # this share of the exploration points is placed within NEAR_R of a wall face
SWEEP_H, SWEEP_PTS = 0.02, 4 # no tunnelling: the segment G(z, t) to G(z, t + SWEEP_H) is tested at SWEEP_PTS points
W_FLOW = 0.5                 # weight of the Fokker Planck consistency of G (see train()): its paths follow v*.
                             # Tested on the fork: without it the generator fans out after the last door and only
                             # 69% of it ends in the zone (agents: 90%); with 0.5, 80%. 0.1 was too weak.
SHARP, W_HAMMER, DEPTH_GAIN = 15.0, C.get("w_hammer", 150.0), 5.0   # cost NEAR a hammer (see hammer()): about 175
HAMMER_R = C.get("hammer_r", 0.4)   # at the rim, fading over about 0.3. Hammers are SOLID disks of radius HAMMER_R
WALL_MARGIN = 0.08           # the steepest part of that cost starts this far outside the rim
W_CONG = 3.0                 # repo: lam_congestion (long range: crowds are unpleasant)
W_PS, PS_R = 10.0, 0.15      # personal space (short range): two people cannot stand on the same spot
W_RUSH = 1.0                 # impatience: running cost W_RUSH * distance to goal, paid at every moment
LAMBDA_HJB = 1.0             # repo: lam_hjb_error
LAMBDA_EXPLORE = 0.5         # weight of the HJB residual at the exploration points (one per batch sample, see train())
REFLECT = False              # False: the SDE agents follow phi and noise, NOTHING else keeps them out of walls or
                             # hammers, so what you see is what phi learned. True adds a physical reflection
                             # (project_out); the metrics always report both.
# Warm start: before the game begins, G is fitted for WARMUP_ITERS steps to "everyone runs straight ahead
# keeping their height" (see train()). No symmetry built in.
WARMUP_ITERS = 2000
WARM_START = "straight"      # "straight": run straight ahead (into obstacles, which then push each agent out on its
                             # own side: this is what makes the crowd split). "stop": wait in front of the first
                             # obstacle (tested: the crowd then collapses into one lane again and starts too slow)
# Mirror symmetry (y -> -y). On a course that is symmetric in y with a monotone congestion cost, the
# Lasry Lions theorem says the equilibrium is UNIQUE, so it must itself be symmetric: a crowd that all
# takes the top lane is a training failure, not a solution. By default the symmetry is NOT imposed:
# the networks must discover it (the "top %" in the snapshot titles measures how well they do).
# mirror=True in a preset imposes it instead (Generator.forward, Phi.forward), as a baseline to compare.
MIRROR = C.get("mirror", False)
FLIP = torch.ones(DIM, device=device)
FLIP[1] = -1.0                               # reflection R: (x1, x2, x3, ...) to (x1, -x2, x3, ...)
G_CUTS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)   # levels of the starting height (in std) where G may tear the crowd
N_PARTNERS = 16              # partners per agent in the congestion estimate (repo: 1). More partners
                             # means less Monte Carlo noise in f, which lowers the HJB residual floor

# ===================== APAC-Net training setup (repo defaults) =====================
BATCH, WIDTH, HH = int(os.environ.get("BATCH", "1024")), 100, 0.5   # repo uses batch 50
ITERS = int(os.environ.get("ITERS", C.get("iters", 25000)))   # repo runs up to 5e5 epochs (ITERS=... for a quick try).
                             # 25000 is enough on every course now: the fork needed 45000 while the walls were
                             # cheap enough to pay, because the lane split was then decided by noise
LR_PHI, LR_G, BETAS = 4e-4, 1e-4, (0.5, 0.9)
LR_DECAY = True                           # linear decay over the last 30% of iterations
WD_PHI = 1e-4                             # repo weight decay
WD_G = C.get("wd_g", 1e-4)                # 0 on split courses: weight decay keeps G smooth, but splitting a
                                          # crowd in two needs a STEEP map near the dividing line
OUT_DIR = f"out_{COURSE}" + (f"_d{DIM}" if DIM != 2 else "") + (f"_seed{SEED}" if SEED else "")   # own folder per course (and per seed)
CKPT = os.path.join(OUT_DIR, "model.pt")  # delete this file to retrain from scratch
os.makedirs(OUT_DIR, exist_ok=True)

def pad(v, fill):
    """Extend a 2D vector with `fill` in the extra (DIM - 2) coordinates."""
    return torch.tensor(list(v) + [fill] * (DIM - 2), dtype=torch.float32, device=device)

MU0, STD0 = pad(START, 0.0), pad(START_STD, 0.3)
GOAL_T = torch.tensor(GOAL, device=device)

# ============================== neural networks ==============================
class ResNet(nn.Module):
    """Repo architecture (DiscNet and GenNet): three hidden layers of width 100,
    s1 = act(W1 [t - T/2, x]),  s2 = act(s1 + hh W2 s1),  s3 = act(s2 + hh W3 s2),  out = W_last s3."""
    def __init__(self, d_out, act, d_in=DIM):
        super().__init__()
        self.lin1 = nn.Linear(d_in + 1, WIDTH)
        self.lin2, self.lin3 = nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, WIDTH)
        self.linlast = nn.Linear(WIDTH, d_out)
        self.act = act

    def forward(self, x, t):
        s = self.act(self.lin1(torch.cat([t - T / 2, x], dim=1)))   # time is centred around 0
        s = self.act(s + HH * self.lin2(s))
        s = self.act(s + HH * self.lin3(s))
        return self.linlast(s)

class Generator(nn.Module):
    """G_theta(z, t) = (1 - t/T) z + (t/T) N_theta(z, t), ReLU like the repo. G(z, 0) = z, so the
    push forward G(., t)#rho_0 starts exactly at rho_0 and conserves mass by construction.
    IMPORTANT: APAC-Net only asks that G(., t)#rho_0 = rho(., t) at every t SEPARATELY. Nothing links
    G(z, t) to G(z, t') for the same z, so t maps to G(z, t) is NOT an agent's trajectory: G may relabel
    who is where between two times. The one thing added here is that such a path may not cut through a
    wall (tunnel_cost in train()): mass cannot pass through a wall, whatever the labels.
    Agents move with the control v* = -grad_p H(grad phi): see simulate_sde.
    Tearing: a ReLU network is continuous in z, and a continuous map cannot split one crowd into two groups
    around an obstacle without leaving a thread of points that connects them. Those points used to sit
    inside the obstacle; now that mass may not be there, they lagged behind it and then snapped back to
    the crowd (rubber banding). So the network also gets the step features 1[z_2 > q] for a few fixed
    levels q of the starting height (G_CUTS, in standard deviations): with them it CAN jump across such a
    level. Which levels it uses, if any, is learned; the levels are not placed on the obstacles."""
    def __init__(self):
        super().__init__()
        self.N = ResNet(DIM, torch.relu, d_in=DIM + len(G_CUTS))   # the repo generator, plus the step inputs
        self.cuts = torch.tensor([G_CUTS], device=device)

    def net(self, z, t):
        z_normalised = (z - MU0) / STD0          # repo GenNet normalises the noise before the network
        steps = (z_normalised[:, 1:2] > self.cuts).float()             # (B, len(G_CUTS)), entries 0 or 1
        return (1 - t / T) * z + (t / T) * self.N(torch.cat([z_normalised, steps], 1), t)

    def forward(self, z, t):
        if not MIRROR:
            return self.net(z, t)
        # Half plane construction: the network only ever sees the TOP half of the starting crowd
        # (z_2 >= 0); a bottom agent is mirrored up, moved by the network, and mirrored back:
        #   G(z, t) = S net(S z, t),   S = R if z_2 < 0, else identity.
        # This is still equivariant, G(Rz, t) = R G(z, t), but unlike averaging it may jump at z_2 = 0.
        # rho_0 puts zero mass on that line, so the push forward is unaffected, and the crowd can be
        # cut cleanly in two: before, a continuous G had to drag a thin strip of agents through the
        # island or pillar to connect the two halves (the grey "tear" agents in the plots).
        S = torch.where(z[:, 1:2] < 0, FLIP, torch.ones_like(FLIP))     # (B, DIM), entries +1 or -1
        return S * self.net(S * z, t)

def g(x):
    """Terminal cost: Huber distance to the goal. Far away it grows like the distance (repo psi_func, slope
    W_GOAL); within GOAL_ZONE_R it is quadratic, so the pull weakens smoothly to zero at the star."""
    d = torch.sqrt(((x[:, :2] - GOAL_T) ** 2).sum(1, keepdim=True) + 1e-8)
    r = GOAL_ZONE_R
    return W_GOAL * torch.where(d < r, d ** 2 / (2 * r), d - r / 2)

class Phi(nn.Module):
    """phi_omega(x, t) = (1 - t/T) N_omega(x, t) + (t/T) g(x) + a_omega * depth_inside_obstacle(x, t).
    The first two terms are the repo ansatz (tanh network, needs 2nd derivatives; the terminal cost g is
    built in). The third is the same idea for the obstacles: inside a wall the HJB reads H(grad phi) = W_IN
    up to small terms, and its solution is 'slope a times the depth, a = H^-1(W_IN)', the cost of the
    fastest way out. a starts at that value and stays trainable (a = A0 exp(log_a)). The term is zero
    outside obstacles, so phi(x, T) = g(x) still holds wherever an agent can be, and N_omega learns
    everything that happens outside them. (Ruthotto et al. do the same: network plus a few explicit terms.)"""
    def __init__(self):
        super().__init__()
        self.N = ResNet(1, torch.tanh)
        lo, hi = 0.0, 1.0e4                      # A0 = H^-1(W_IN) by bisection (H is increasing in |p|)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            e1 = torch.zeros(1, DIM, device=device)
            e1[0, 0] = mid
            lo, hi = (mid, hi) if ham(e1).item() < W_IN else (lo, mid)
        self.A0 = lo
        self.log_a = nn.Parameter(torch.zeros(1))

    def base(self, x, t):
        """The repo ansatz: network plus built in terminal cost."""
        n = self.N(x, t)
        if MIRROR:                               # invariant version phi(Rx, t) = phi(x, t)
            n = 0.5 * (n + self.N(FLIP * x, t))
        return (1 - t / T) * n + (t / T) * g(x)

    def slope(self):
        return self.A0 * torch.exp(self.log_a)

    def forward(self, x, t):
        return self.base(x, t) + self.slope() * inside_depth(x, t)

# ========================= Hamiltonian and running costs =========================
def ham(p):
    """quadratic:   H(p) = |p|^2 / (2 effort), the Legendre transform of L(v) = effort |v|^2 / 2.
    capped:      the Legendre transform of L(v) = effort |v|^2 / 2 for |v| <= VMAX (faster impossible):
                 H = |p|^2 / (2 effort) while |p| <= effort VMAX, then VMAX |p| - effort VMAX^2 / 2.
                 Same value and slope at the switch, so H is smooth enough for the HJB residual.
    speed_limit: H(p) = c|p| (repo bottleneck env), smoothed as c (sqrt(|p|^2 + eps^2) - eps)."""
    if HAM == "quadratic":
        return (p ** 2).sum(1, keepdim=True) / (2 * EFFORT)
    if HAM == "soft_cap":
        # Legendre transform of L(v) = effort v^2 / 2 up to VMAX, then the extra speed costs TAIL times more:
        # H = n^2 / (2 effort) for n <= pc, else pc^2 / (2 effort) + VMAX (n - pc) + (n - pc)^2 / (2 TAIL effort)
        n = torch.sqrt((p ** 2).sum(1, keepdim=True) + 1e-12)
        pc = EFFORT * VMAX
        over = torch.relu(n - pc)
        return torch.where(n <= pc, n ** 2 / (2 * EFFORT),
                           pc ** 2 / (2 * EFFORT) + VMAX * over + over ** 2 / (2 * TAIL * EFFORT))
    if HAM == "capped":
        n = torch.sqrt((p ** 2).sum(1, keepdim=True) + 1e-12)
        pc = EFFORT * VMAX                                   # |p| at which the top speed is reached
        return torch.where(n <= pc, n ** 2 / (2 * EFFORT), VMAX * n - EFFORT * VMAX ** 2 / 2)
    return SPEED * (torch.sqrt((p ** 2).sum(1, keepdim=True) + EPS_H ** 2) - EPS_H)

def velocity(p):
    """Optimal feedback v* = -grad_p H(grad phi): -grad phi / effort (quadratic), or top speed c in the
    direction of steepest descent of phi (speed_limit)."""
    if HAM == "quadratic":
        return -p / EFFORT
    if HAM == "soft_cap":                                    # speed n/effort up to VMAX, then slowly above it
        n = torch.sqrt((p ** 2).sum(1, keepdim=True) + 1e-12)
        speed = torch.where(n <= EFFORT * VMAX, n / EFFORT, VMAX + (n - EFFORT * VMAX) / (TAIL * EFFORT))
        return -p / n * speed
    if HAM == "capped":                                      # -grad phi / effort, but never faster than VMAX
        n = torch.sqrt((p ** 2).sum(1, keepdim=True) + 1e-12)
        return -p / EFFORT * torch.clamp(EFFORT * VMAX / n, max=1.0)
    return -SPEED * p / torch.sqrt((p ** 2).sum(1, keepdim=True) + EPS_H ** 2)

BOXES = torch.tensor([b[:4] for b in C["boxes"]], device=device)     # (n_boxes, 4): x0, x1, y0, y1

def box_depths(x):
    """Signed depth inside every box: distance to the nearest edge (positive inside), shape (B, n_boxes)."""
    X, Y = x[:, 0:1], x[:, 1:2]                                       # (B, 1) against (n_boxes,): all boxes at once
    return torch.minimum(torch.minimum(X - BOXES[:, 0], BOXES[:, 1] - X), torch.minimum(Y - BOXES[:, 2], BOXES[:, 3] - Y))

def inside_depth(x, t):
    """How deep a point is inside an obstacle: distance to the nearest edge of a wall box, or to the rim of a
    hammer. 0 outside every obstacle. Shape (B, 1). This is the shape of phi's explicit obstacle term."""
    d = torch.relu(box_depths(x)).sum(1, keepdim=True)
    if C["hammers"]:
        d = d + torch.relu(HAMMER_R - hammer_dist(x, t)).sum(0)
    return d

def inside_depth_derivs(x, t):
    """Gradient in x and derivative in t of inside_depth (the hammers move, so the depth changes in time).
    Inside a wall the gradient is the unit vector pointing deeper in; everything is zero outside obstacles."""
    xd, td = x.detach().requires_grad_(True), t.detach().requires_grad_(True)
    gx, gt = torch.autograd.grad(inside_depth(xd, td).sum(), (xd, td), allow_unused=True)
    return gx, (torch.zeros_like(td) if gt is None else gt)

def inside_any(x, t):
    """1.0 where a point is inside a wall box or a hammer as drawn, else 0.0, shape (B, 1)."""
    return (obstacle_depth(x, t) > 0).float()

def in_box(x):
    """1.0 where a point is inside a wall box as drawn, else 0.0, shape (B, 1)."""
    return (box_depths(x) > 0).any(1, keepdim=True).float()

def walls(x):
    """Wall cost in f: W_IN per unit time inside a box, and outside it the skirt W_SKIRT (1 - dist / R_SKIRT)^2
    within R_SKIRT of a face (W_SKIRT at the face, zero from R_SKIRT on)."""
    skirt = W_SKIRT * (torch.relu(1 + box_depths(x) / R_SKIRT).clamp(max=1.0) ** 2).sum(1, keepdim=True)
    return torch.where(in_box(x) > 0, torch.full_like(skirt, W_IN), skirt)

def push_out(x, t):
    """Population only: generator mass inside an obstacle pays W_OFF + W_PUSH * depth (times the box
    multiplier). The HJB cost W_IN is flat inside an obstacle and phi prices it, so by itself it gives G no
    direction; this term grows with depth, so its gradient points to the nearest way out. (W_OFF keeps the
    total positive inside a moving hammer, where phi's explicit term is only approximate.)"""
    d = (BOX_W * torch.relu(box_depths(x))).sum(1, keepdim=True)
    if C["hammers"]:
        d = d + torch.relu(HAMMER_R - hammer_dist(x, t)).sum(0)
    return W_PUSH * d + W_OFF * inside_any(x, t)

def tunnel_cost(xa, xb):
    """No tunnelling for the generator. xa = G(z, t) and xb = G(z, t + h) are two points of the SAME path.
    If the straight segment between them cuts through a box, the path jumped through a wall. Each test
    point on the segment that lies inside a box costs W_PUSH (1 / DEPTH_GAIN + distance to the nearest END
    of that wall, measured across the direction of travel). The gradient therefore slides the crossing
    point ALONG the wall to the nearest opening: the path is rerouted through the door, not shoved out on
    the far side. (G is continuous in t, so 'no path crosses a wall' is the zero flux wall condition of
    the Fokker Planck equation, imposed on the generator directly.)"""
    d = (xb - xa)[:, :2]
    along_x = d[:, 0:1].abs() >= d[:, 1:2].abs()                   # travelling mainly in x: escape sideways, in y
    cost = torch.zeros(xa.shape[0], 1, device=xa.device)
    for s in torch.rand(SWEEP_PTS, xa.shape[0], 1, device=xa.device):   # random points on every segment
        p = xa + s * (xb - xa)
        X, Y = p[:, 0:1], p[:, 1:2]
        for b in C["boxes"]:
            x0, x1, y0, y1 = b[:4]
            inside = ((X > x0) & (X < x1) & (Y > y0) & (Y < y1)).float()
            side = torch.where(along_x, torch.minimum(Y - y0, y1 - Y), torch.minimum(X - x0, x1 - X))
            cost = cost + inside * (1 + DEPTH_GAIN * side)
    return W_PUSH / DEPTH_GAIN * cost / SWEEP_PTS

def _faces():
    """Every box edge inside the map as a row (start x, start y, end x, end y, outward normal x, y)."""
    rows = []
    for b in C["boxes"]:
        x0, x1, y0, y1 = b[:4]
        ya, yb, xa, xb = max(y0, YL[0]), min(y1, YL[1]), max(x0, XL[0]), min(x1, XL[1])
        rows += [(x0, ya, x0, yb, -1.0, 0.0)] * (x0 > XL[0]) + [(x1, ya, x1, yb, 1.0, 0.0)] * (x1 < XL[1])
        rows += [(xa, y0, xb, y0, 0.0, -1.0)] * (y0 > YL[0]) + [(xa, y1, xb, y1, 0.0, 1.0)] * (y1 < YL[1])
    return torch.tensor(rows, dtype=torch.float32, device=device)

BOX_W = torch.tensor([[b[4] if len(b) > 4 else 1.0 for b in C["boxes"]]], device=device)   # per box multiplier
FACES = _faces()
FACE_LEN = (FACES[:, 2:4] - FACES[:, 0:2]).norm(dim=1)

def sample_faces(n):
    """n points on the wall faces (a hair outside the wall), chosen uniformly by length, with outward normals."""
    f = FACES[torch.multinomial(FACE_LEN, n, replacement=True)]
    p = sample_rho0(n)                                             # extra coordinates like rho_0
    p[:, :2] = f[:, 0:2] + torch.rand(n, 1, device=device) * (f[:, 2:4] - f[:, 0:2]) + 0.01 * f[:, 4:6]
    return p, f[:, 4:6]

def hammer_centres(t):
    """Centres of all hammers at times t, shape (len(hammers), B, 2). This makes f depend on time."""
    if not C["hammers"]:
        return torch.zeros(0, t.shape[0], 2, device=t.device)
    return torch.stack([torch.cat([torch.full_like(t, hx), hy + amp * torch.sin(2 * np.pi * fr * (t - t0))], 1)
                        for hx, hy, amp, fr, t0 in C["hammers"]])

def hammer_dist(x, t):
    """Distance from every point to every hammer centre, shape (H, B, 1)."""
    return torch.sqrt(((x[None, :, :2] - hammer_centres(t)) ** 2).sum(-1, keepdim=True) + 1e-8)

def hammer(x, t):
    """Hammer cost in f. Inside the disk: W_IN per unit time, like a wall. Outside: a smooth step that starts
    WALL_MARGIN before the rim plus a ramp, about W_HAMMER * 1.2 at the rim and fading over about 0.3, so the
    value function learns to keep clear of where the hammer is and is going to be."""
    if not C["hammers"]:
        return torch.zeros_like(t)
    depth = HAMMER_R - hammer_dist(x, t)                               # (H, B, 1), positive inside
    near = W_HAMMER * (torch.sigmoid(SHARP * (depth + WALL_MARGIN)) + DEPTH_GAIN * torch.relu(depth + WALL_MARGIN))
    near = torch.where(depth > 0, torch.zeros_like(near), near).sum(0)
    return torch.where((depth > 0).any(0), torch.full_like(near, W_IN), near)

def obstacle_depth(x, t):
    """How deep each point is inside the nearest obstacle AS DRAWN (no margin): > 0 inside, < 0 outside."""
    depth = box_depths(x).max(1, keepdim=True)[0]
    if C["hammers"]:
        depth = torch.maximum(depth, (HAMMER_R - hammer_dist(x, t)).max(0)[0])
    return depth

@torch.no_grad()
def partners(G, t):
    """N_PARTNERS independent samples of rho_t for every agent, at the SAME times t: (N_PARTNERS, B, DIM).
    No gradient: the partners are 'the crowd', a given density that each agent reacts to."""
    B = t.shape[0]
    return G(sample_rho0(N_PARTNERS * B), t.repeat(N_PARTNERS, 1)).view(N_PARTNERS, B, DIM)

def congestion(x, x2):
    """Repo estimator: x2 holds INDEPENDENT samples of rho_t at the same times t as x, and
    K(x - x2) = 1 / (1 + |x - x2|^2). Averaging over x2 gives (K * rho_t)(x), so a few extra
    generator calls replace the N x N interaction matrix. The repo uses one partner; we average several."""
    d2 = ((x[None, :, :2] - x2[:, :, :2]) ** 2).sum(-1, keepdim=True)          # (N_PARTNERS, B, 1)
    crowd = W_CONG / (1 + d2)                                                   # repo kernel, long range
    space = W_PS * torch.exp(-d2 / (2 * PS_R ** 2))                             # personal space, short range
    # Both kernels have a positive Fourier transform, so f stays MONOTONE: the Lasry Lions uniqueness
    # (and the mirror symmetry argument) still holds. A hard "max density" constraint would be the
    # exact model of "people cannot overlap"; this steep repulsion is its smooth, trainable version.
    return (crowd + space).mean(0)

def rush(x):
    """Impatience: every moment away from the goal costs W_RUSH * (Huber) distance. Without it only the
    position at time T counts, so waiting is free; with it, getting there early pays off."""
    return (W_RUSH / W_GOAL) * g(x)

def sample_rho0(n):
    return MU0 + STD0 * torch.randn(n, DIM, device=device)

def sample_explore(n):
    """Exploration points for the HJB residual: uniform over the map, except a share NEAR_WALL of them placed
    within NEAR_R of a wall face (outside the wall), because that is where phi is hardest to get right and
    where a wrong slope shows at once. Extra coordinates are drawn like rho_0."""
    x = sample_rho0(n)
    x[:, 0] = XL[0] + (XL[1] - XL[0]) * torch.rand(n, device=device)
    x[:, 1] = YL[0] + (YL[1] - YL[0]) * torch.rand(n, device=device)
    k = int(NEAR_WALL * n)
    p, normal = sample_faces(k)
    x[:k, :2] = p[:, :2] + NEAR_R * torch.rand(k, 1, device=device) * normal
    return x

def sample_t(n):
    return T * torch.rand(n, 1, device=device)

# ================= HJB operator via autograd (repo get_hjb_loss + get_trace) =================
def hjb_terms(phi, x, t):
    """Returns d_t phi + nu Lap phi - H(grad phi) at (x, t) and grad phi, differentiable in omega and in x.
    Laplacian trick from the repo: stack DIM copies of the batch; copy k is contracted with e_k, so
    ONE autograd call gives row k of the Hessian for copy k, and we keep its diagonal entry.
    phi's obstacle term (slope times depth inside an obstacle) is piecewise linear in x: it adds
    slope * (its gradient) to grad phi and slope * (its time derivative, for a moving hammer) to d_t phi,
    both added at the end, which keeps the autograd passes as cheap as for the plain repo ansatz. (Its
    Laplacian is zero inside a wall and -1 / distance to the centre inside a hammer; the latter is dropped,
    it is under 1% of W_IN and only matters inside a hammer.)"""
    B = x.shape[0]
    x_rep = x.repeat(DIM, 1)                                   # (DIM B, DIM)
    t_rep = t.repeat(DIM, 1).requires_grad_(True)              # fresh leaf: partial d_t phi, not through G
    out = phi.base(x_rep, t_rep)                               # (DIM B, 1)
    phi_x, phi_t = torch.autograd.grad(out.sum(), (x_rep, t_rep), create_graph=True)
    E = torch.eye(DIM, device=device).repeat_interleave(B, dim=0)       # block k of rows holds e_k
    hess_rows = torch.autograd.grad(phi_x, x_rep, grad_outputs=E, create_graph=True)[0]  # e_k^T Hess phi
    lap = (hess_rows * E).sum(1).view(DIM, B).sum(0).unsqueeze(1)       # sum_k d^2 phi / dx_k^2, (B, 1)
    d_x, d_t = inside_depth_derivs(x, t)
    grad_phi = phi_x[:B] + phi.slope() * d_x                            # gradient of the full phi
    return phi_t[:B] + phi.slope() * d_t + NU * lap - ham(grad_phi), grad_phi

def set_requires_grad(net, flag):
    for p in net.parameters():
        p.requires_grad_(flag)

# ================================ training ================================
def train():
    G, phi = Generator().to(device), Phi().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS, weight_decay=WD_G)
    opt_phi = torch.optim.Adam(phi.parameters(), lr=LR_PHI, betas=BETAS, weight_decay=WD_PHI)
    # Learning rates: constant for the first 70% (the game must be free to switch strategy), then a
    # linear decay to 10% so both players settle on the equilibrium. Not in the repo; set LR_DECAY = False.
    decay = lambda it: 1.0 if (not LR_DECAY or it < 0.7 * ITERS) else 1.0 - 0.9 * (it - 0.7 * ITERS) / (0.3 * ITERS)
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, decay)
    sched_phi = torch.optim.lr_scheduler.LambdaLR(opt_phi, decay)
    hist = {"loss_phi": [], "loss_G": [], "HJB residual": []}
    buf = []                                          # GPU side buffer for the history (see end of the loop)

    # ===== Warm start of G (not in the repo) =====
    # Seed 0 sent the whole crowd over the top, seed 1 the whole crowd underneath: the lane choice was
    # made by the random initial weights, in the first few hundred iterations, and then never undone
    # (an agent can only change lane by being pushed through the island). Starting instead from the
    # neutral plan "run straight ahead, keeping your height", every agent meets the island on its own side
    # of the centre line, and the island pushes it out towards that side. (With the old HARD speed cap this
    # start also set off the runaway; with the soft cap the saddle problem stays well posed.)
    # The same rule is used on every course.
    x_stop = min(b[0] for b in C["boxes"]) - 0.25       # just in front of the leftmost obstacle
    opt_warm = torch.optim.Adam(G.parameters(), lr=1e-3)
    t_warm = time.time()
    for it in range(WARMUP_ITERS):
        if it % 10 == 0 or it == WARMUP_ITERS - 1:
            progress(it + 1, WARMUP_ITERS, t_warm, "warm start")
        z, t = sample_rho0(BATCH), sample_t(BATCH)
        target = z.clone()
        straight = z[:, 0] + (t[:, 0] / T) * (GOAL[0] - z[:, 0])         # straight ahead, same height
        if WARM_START == "stop":
            straight = torch.minimum(straight, torch.maximum(z[:, 0], torch.full_like(straight, x_stop)))
        target[:, 0] = straight
        loss_warm = ((G(z, t) - target) ** 2).mean()
        opt_warm.zero_grad(), loss_warm.backward(), opt_warm.step()
    log(f"warm start done, fit error {loss_warm.item():.4f}", fresh=True)
    t_start = time.time()
    for it in range(ITERS + 1):
        if it % 10 == 0 or it == ITERS:
            progress(it, ITERS, t_start)
        # ===== (1) phi step (discriminator): maximise the Lagrangian over omega =====
        set_requires_grad(phi, True), set_requires_grad(G, False)
        z, t = sample_rho0(BATCH), sample_t(BATCH)
        with torch.no_grad():
            x = G(z, t)                                     # x ~ rho(., t)
            x2 = partners(G, t)                             # independent copies at the same t
            # Exploration points: the HJB must hold EVERYWHERE an agent can be, not only where the crowd is
            # (noise knocks SDE agents off the streams, and they must still be steered sensibly). One per
            # batch sample and at the same time t, so the same partners give the crowd cost there.
            xe = sample_explore(BATCH)
            crowd = lambda y: rush(y) + congestion(y, x2)
            FF = walls(x) + hammer(x, t) + crowd(x)                          # f(x, rho, t) at the crowd
            FFe = walls(xe) + hammer(xe, t) + crowd(xe)                      # and at the exploration points
            w, we = IN_WEIGHT / (IN_WEIGHT + W_IN * inside_any(x, t)), IN_WEIGHT / (IN_WEIGHT + W_IN * inside_any(xe, t))
            xf, normal = sample_faces(N_FACE)               # points on the wall faces, with outward normals
            tf = sample_t(N_FACE)
        # one autograd pass for the three sets of points (a third of the GPU calls of three separate passes)
        pts = torch.cat([x, xe, xf]).requires_grad_(True)
        lhs_all, grad_all = hjb_terms(phi, pts, torch.cat([t, t, tf]))
        lhs, lhs_e, grad_f = lhs_all[:BATCH], lhs_all[BATCH:2 * BATCH], grad_all[2 * BATCH:]
        phi0 = phi(sample_rho0(BATCH), torch.zeros(BATCH, 1, device=device)).mean()  # E_{rho_0}[phi(., 0)]
        hjb_res = (w * (lhs + FF).abs()).mean()             # |d_t phi + nu Lap phi - H + f|, everywhere
        explore_res = (we * (lhs_e + FFe).abs()).mean()
        # Wall face condition: just outside a wall face the control v = -grad_p H(grad phi) must not point
        # INTO the wall, i.e. grad phi . n <= 0 for the outward normal n (one sided: phi may still push agents
        # AWAY from the wall, and the skirt cost makes it do so). Inside the wall phi rises steeply, so a phi
        # that falls towards the face from outside would make a pit at the face that agents run into from both
        # sides. The exact HJB solution has no such pits, but they are invisible at sample points. Faded out
        # towards T, where phi = g is fixed by the ansatz.
        face = ((1 - tf / T) * torch.relu((grad_f[:, :2] * normal).sum(1, keepdim=True))).mean()
        loss_phi = (-(phi0 + T * (w * lhs).mean()) + LAMBDA_HJB * hjb_res + LAMBDA_EXPLORE * explore_res
                    + LAMBDA_FACE * face)
        opt_phi.zero_grad(), loss_phi.backward(), opt_phi.step()

        # ===== (2) generator step: minimise the Lagrangian over theta =====
        set_requires_grad(phi, False), set_requires_grad(G, True)
        z, t = sample_rho0(BATCH), sample_t(BATCH)
        x, x2 = G(z, t), partners(G, t)                     # x keeps its graph back to theta, the partners do not
        lhs, grad_phi = hjb_terms(phi, x, t)                # phi's derivatives at x = G(z, t)
        # Crowd cost: f = dF/drho for F(rho) = W/2 int int K rho rho is W (K * rho)(x), the crowd being a
        # GIVEN density, so the gradient flows through x only. (Same expected gradient as half the cost with
        # gradients through x AND the partners, at a third of the work. The repo sends gradients through both
        # at full weight, which doubles the congestion weight for G only.)
        lagrangian = lhs + walls(x) + hammer(x, t) + rush(x) + congestion(x, x2)   # the same f as in the HJB
        # plus the population's own penalty for mass inside an obstacle (see push_out)
        loss_G = T * (lagrangian + push_out(x, t)).mean()
        # and no generator path may jump through a wall between t and t + SWEEP_H (see tunnel_cost)
        t_next = torch.clamp(t + SWEEP_H, max=T)
        x_next = G(z, t_next)
        loss_G = loss_G + T * tunnel_cost(x, x_next).mean()
        # Fokker Planck consistency (W_FLOW): the population must MOVE the way its agents move. Between t and
        # t + SWEEP_H a generator path should advance by v*(x, t) dt, with v* = -grad_p H(grad phi) the agents'
        # optimal velocity. It is the mirror image of APAC-Net's HJB penalty on phi: that one keeps the value
        # function on the HJB, this one keeps the population on the flow that the value function prescribes.
        v_star = velocity(grad_phi.detach())
        flow = (x_next - x - v_star * (t_next - t)) / SWEEP_H
        loss_G = loss_G + W_FLOW * T * (flow ** 2).sum(1).mean()
        opt_G.zero_grad(), loss_G.backward(), opt_G.step()
        sched_G.step(), sched_phi.step()

        # .item() forces the CPU to wait for the GPU. Doing it 3 times per iteration stalled the GPU
        # every step; now the values stay on the GPU and are copied over once every 500 iterations.
        buf.append(torch.stack([loss_phi.detach(), loss_G.detach(), hjb_res.detach()]))
        if it % 500 == 0 or it == ITERS:
            vals = torch.stack(buf).cpu().tolist()
            buf = []
            for row in vals:
                for k, v in zip(hist, row):
                    hist[k].append(v)
            lp, lg, hr = vals[-1]
            train.info = dict(device=str(device), it_per_s=it / max(time.time() - t_start, 1e-9))
            log(f"it {it:6d} | loss_phi {lp:8.3f} | loss_G {lg:8.3f} | HJB residual {hr:7.4f} | {train.info['it_per_s']:5.1f} it/s")
    torch.save({"G": G.state_dict(), "phi": phi.state_dict(), "hist": hist, "info": train.info}, CKPT)
    return G, phi, hist

# ================================ visualisation ================================

@torch.no_grad()
def gen_at(G, z, t):
    return G(z, torch.full((z.shape[0], 1), float(t), device=device)).cpu()

def project_out(X, t):
    """Reflecting boundary for the SDE (Skorokhod projection): an agent that ends a step inside a box is
    put back just outside its nearest edge, one inside a hammer just outside its rim. This is the particle
    picture of a hard wall with zero flux (Neumann condition) in the PDE. Returns X and a mask of who
    was moved, so the number of wall contacts can be reported honestly instead of hidden."""
    moved = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
    x, y = X[:, 0], X[:, 1]
    for b in C["boxes"]:
        x0, x1, y0, y1 = b[:4]
        inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        if inside.any():
            edge = torch.stack([x - x0, x1 - x, y - y0, y1 - y], 1).argmin(1)    # nearest edge
            x = torch.where(inside & (edge == 0), torch.full_like(x, x0 - 1e-3), x)
            x = torch.where(inside & (edge == 1), torch.full_like(x, x1 + 1e-3), x)
            y = torch.where(inside & (edge == 2), torch.full_like(y, y0 - 1e-3), y)
            y = torch.where(inside & (edge == 3), torch.full_like(y, y1 + 1e-3), y)
            moved |= inside
    for c in hammer_centres(torch.full((X.shape[0], 1), float(t), device=X.device)):   # (B, 2) per hammer
        d = torch.stack([x, y], 1) - c
        r = d.norm(dim=1, keepdim=True).clamp_min(1e-6)
        inside = (r[:, 0] < HAMMER_R)
        push = c + d / r * (HAMMER_R + 1e-3)                                       # onto the rim
        x = torch.where(inside, push[:, 0], x)
        y = torch.where(inside, push[:, 1], y)
        moved |= inside
    for b in C["boxes"] if C["hammers"] else []:       # a hammer may have pushed someone into a wall
        x0, x1, y0, y1 = b[:4]
        inside = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        if inside.any():
            edge = torch.stack([x - x0, x1 - x, y - y0, y1 - y], 1).argmin(1)
            x = torch.where(inside & (edge == 0), torch.full_like(x, x0 - 1e-3), x)
            x = torch.where(inside & (edge == 1), torch.full_like(x, x1 + 1e-3), x)
            y = torch.where(inside & (edge == 2), torch.full_like(y, y0 - 1e-3), y)
            y = torch.where(inside & (edge == 3), torch.full_like(y, y1 + 1e-3), y)
    X = X.clone()
    X[:, 0], X[:, 1] = x, y
    return X, moved

MAX_MOVE = 0.08                  # largest drift move per SDE sub step (see simulate_sde)

def simulate_sde(phi, z, n_steps, noise=True, reflect=None):
    """Euler Maruyama for dX = v*(X, t) dt + sqrt(2 nu) dW using ONLY phi. Independent check of the
    saddle point: these noisy agents should reproduce the generator's crowd at every time.
    noise=False drops the dW term: each agent's planned route under the learned control.
    reflect (default REFLECT): False = nothing but phi keeps the agents out of obstacles; True = an agent
    that ends a step inside one is put back outside (project_out).
    simulate_sde.contacts = % of agents that were inside an obstacle at some moment (with reflect: that
    needed the reflection); simulate_sde.max_depth = the deepest any agent got."""
    reflect = REFLECT if reflect is None else reflect
    # Sub steps of at most 0.004, and each drift move clipped to MAX_MOVE (under half the thinnest wall),
    # so even a wrong control cannot carry an agent across a wall in one step. The clip is a numerical
    # safeguard: it only bites above MAX_MOVE / dt (about 20), far above normal running speeds.
    sub = max(1, int(np.ceil((T / n_steps) / 0.004)))
    dt, X, path = T / (n_steps * sub), z.clone(), [z.cpu()]
    touched = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    deepest = 0.0
    for k in range(n_steps * sub):
        Xg = X.detach().requires_grad_(True)
        p = torch.autograd.grad(phi(Xg, torch.full((X.shape[0], 1), k * dt, device=device)).sum(), Xg)[0]
        move = velocity(p) * dt
        mn = move[:, :2].norm(dim=1, keepdim=True)
        move = move * torch.clamp(MAX_MOVE / (mn + 1e-12), max=1.0)
        X = (X + move + (np.sqrt(2 * NU * dt) * torch.randn_like(X) if noise else 0)).detach()
        depth = obstacle_depth(X, torch.full((X.shape[0], 1), (k + 1) * dt, device=device))[:, 0]
        touched |= depth > 0
        deepest = max(deepest, depth.max().item())
        if reflect:
            X, _ = project_out(X, (k + 1) * dt)
        if (k + 1) % sub == 0:
            path.append(X.detach().cpu())
    simulate_sde.contacts = 100 * touched.float().mean().item()
    simulate_sde.max_depth = deepest
    return torch.stack(path)                                 # (n_steps + 1, n, DIM)

def illegal(pts, t):
    """% of points inside an obstacle as DRAWN: inside a box, or inside a hammer disk."""
    x = pts.to(device)
    bad = (box_depths(x) > 0).any(1)
    if C["hammers"]:
        bad |= (hammer_dist(x, torch.full((x.shape[0], 1), float(t), device=device)) < HAMMER_R).any(0)[:, 0]
    return 100 * bad.float().mean().item()

def tunnelling(frames):
    """% of paths (same agent or same z over consecutive frames) that cut through a wall box at least once:
    8 test points on every frame to frame segment, counted when more than 0.05 deep (a straight segment
    between two frames may clip a corner that the real path went around)."""
    hit = torch.zeros(frames[0].shape[0], dtype=torch.bool, device=device)
    for a, b in zip(frames[:-1], frames[1:]):
        for s in torch.linspace(0.0, 1.0, 8):
            hit |= (box_depths((a + s * (b - a)).to(device)) > 0.05).any(1)
    return 100 * hit.float().mean().item()

def symmetry_error(frames):
    """Mirror symmetry over the whole run: at every frame, among agents clearly off the centre line
    (|y| > 0.1), the top share should be 50%. Returns the average |top share - 50| in percentage points,
    over frames where at least 20% of agents are off the centre line (agents squeezing through a central
    gap tell us nothing about symmetry)."""
    errs = []
    for pts in frames:
        a, b = (pts[:, 1] > 0.1).sum().item(), (pts[:, 1] < -0.1).sum().item()
        if a + b > 0.2 * pts.shape[0]:
            errs.append(abs(a / (a + b) - 0.5) * 100)
    return float(np.mean(errs)) if errs else 0.0

def out_of_bounds(pts):
    """% of points outside the plotted map (agents that wander off, or density sprayed far away)."""
    x, y = pts[:, 0], pts[:, 1]
    return 100 * ((x < XL[0]) | (x > XL[1]) | (y < YL[0]) | (y > YL[1])).float().mean().item()

def qualified(pts):
    return 100 * ((((pts[:, :2] - torch.tensor(GOAL)) ** 2).sum(1).sqrt() < FINISH_R).float().mean().item())

def course_image():
    xs, ys = torch.meshgrid(torch.linspace(*XL, 300), torch.linspace(*YL, 200), indexing="xy")
    pts = torch.zeros(xs.numel(), DIM, device=device)
    pts[:, 0], pts[:, 1] = xs.flatten().to(device), ys.flatten().to(device)
    img = in_box(pts).reshape(xs.shape).cpu().numpy()
    img[img < 0.5] = float("nan")                           # transparent outside the walls
    return img

def draw_course(ax, t, img):
    ax.imshow(img, extent=(*XL, *YL), origin="lower", cmap="Greys", vmin=0, vmax=1, zorder=2)
    for c in hammer_centres(torch.tensor([[float(t)]]))[:, 0]:
        ax.add_patch(plt.Circle((c[0].item(), c[1].item()), HAMMER_R, color="crimson", alpha=0.6, zorder=3))
    ax.add_patch(plt.Circle(GOAL, FINISH_R, fill=False, ls="--", color="goldenrod", zorder=3))
    ax.scatter(*GOAL, marker="*", s=250, c="gold", edgecolors="k", zorder=3)   # below the agents
    ax.set_xlim(*XL), ax.set_ylim(*YL), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])

def save(name, saver):
    """Save into OUT_DIR. On Windows a file that is open in a viewer (or a OneDrive placeholder)
    raises OSError 22, so fall back to a fresh, numbered name instead of crashing."""
    path, k = os.path.join(OUT_DIR, name), 1
    while True:
        try:
            saver(path)
            return print(f"saved {path}")
        except OSError:
            path, k = os.path.join(OUT_DIR, f"{k}_{name}"), k + 1
            if k > 20:
                raise

def progress(done, total, t0, label="training", width=30):
    """One-line progress bar (rewritten in place): [#####-----]  42.0% | 4200/10000 | 12.3 it/s | ETA 00:07:52."""
    if not sys.stdout.isatty():                          # redirected to a file (e.g. console.txt): no bar
        return
    frac = min(max(done / max(total, 1), 0.0), 1.0)
    el = time.time() - t0
    rate = done / el if el > 0 else 0.0
    eta = int((total - done) / rate) if rate > 0 else 0
    bar = "#" * int(width * frac) + "-" * (width - int(width * frac))
    sys.stdout.write(f"\r{label} [{bar}] {100 * frac:5.1f}% | {done}/{total} | {rate:5.1f} it/s | ETA {eta // 3600:02d}:{eta % 3600 // 60:02d}:{eta % 60:02d}  ")
    sys.stdout.flush()

def log(msg, fname="out.txt", fresh=False):
    """Print and append to OUT_DIR/<fname> (out.txt = training log, metrics.txt = collision summary)."""
    if sys.stdout.isatty():
        sys.stdout.write("\r" + " " * 100 + "\r")  # wipe the progress bar line before printing
    print(msg)
    with open(os.path.join(OUT_DIR, fname), "w" if fresh else "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def write_csv(hist):
    """Every iteration of the training history to OUT_DIR/out.csv (replaces the old to_csv.py step)."""
    with open(os.path.join(OUT_DIR, "out.csv"), "w", encoding="utf-8") as f:
        f.write("it," + ",".join(hist) + "\n")
        for it, row in enumerate(zip(*hist.values())):
            f.write(f"{it}," + ",".join(f"{v:.6f}" for v in row) + "\n")

def scatter_agents(ax, pts, col):
    ax.scatter(pts[:, 0], pts[:, 1], c=col, cmap="coolwarm", vmin=-2, vmax=2, s=5, zorder=4)

def plot_training(hist):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (name, v) in zip(axes, hist.items()):
        v, w = np.array(v), min(200, len(v))       # moving average window (shorter for a quick try)
        ax.plot(v, alpha=0.25), ax.plot(np.arange(w - 1, len(v)), np.convolve(v, np.ones(w) / w, "valid"))
        ax.set_title(name), ax.set_xlabel("iteration")
    axes[2].set_yscale("log")
    fig.tight_layout(), save("training.png", lambda p: fig.savefig(p, dpi=150)), plt.show()

def plot_snapshots(G, phi, n=800, fracs=(0.0, 0.25, 0.5, 0.75, 1.0), n_steps=100):
    """Top row: generator G_theta(z, t). Bottom row: SDE agents steered by phi_omega.
    Colour = starting height z_2, so you can see who takes which lane."""
    z = sample_rho0(n)
    col, sde, img = z[:, 1].cpu(), simulate_sde(phi, z, n_steps), course_image()
    fig, axes = plt.subplots(2, len(fracs), figsize=(3.4 * len(fracs), 5.2))
    times = [f * T for f in fracs]                      # snapshot times as fractions of the horizon
    for j, t in enumerate(times):
        for i, pts in enumerate([gen_at(G, z, t), sde[round(t / T * n_steps)]]):
            draw_course(axes[i, j], t, img), scatter_agents(axes[i, j], pts, col)
            top = 100 * (pts[:, 1] > 0).float().mean().item()   # 50% = symmetric split (learned, not imposed)
            axes[i, j].set_title(f"t={t:.2f} | qual {qualified(pts):.0f}% | top {top:.0f}% | "
                                 f"illegal {illegal(pts, t):.1f}%", fontsize=8)
    axes[0, 0].set_ylabel("density: generator $G_\\theta(z,t)$"), axes[1, 0].set_ylabel("agents: SDE, $v=-\\nabla_pH(\\nabla\\phi_\\omega)$")
    ts = [k * T / n_steps for k in range(n_steps + 1)]
    contacts, deepest = simulate_sde.contacts, simulate_sde.max_depth   # from the simulate_sde call above
    torch.manual_seed(SEED + 1)
    other = simulate_sde(phi, z, n_steps, reflect=not REFLECT)          # same agents, the other setting of REFLECT
    other_contacts, other_qual = simulate_sde.contacts, qualified(other[n_steps])
    gen = [gen_at(G, z, ts[k]) for k in range(n_steps + 1)]
    ill_sde = [illegal(sde[k], ts[k]) for k in range(n_steps + 1)]
    ill_gen = [illegal(gen[k], ts[k]) for k in range(n_steps + 1)]
    oob_sde = [out_of_bounds(sde[k]) for k in range(n_steps + 1)]
    oob_gen = [out_of_bounds(gen[k]) for k in range(n_steps + 1)]
    # speeds between consecutive frames (same agent / same z), and time spent hugging an obstacle
    dt_f = T / n_steps
    spd_sde = torch.stack([(sde[k + 1] - sde[k])[:, :2].norm(dim=1) / dt_f for k in range(n_steps)])
    spd_gen = torch.stack([(gen[k + 1] - gen[k])[:, :2].norm(dim=1) / dt_f for k in range(n_steps)])
    jerk_sde = (spd_sde[1:] - spd_sde[:-1]).abs().median().item()       # typical frame to frame speed change
    near = [(obstacle_depth(sde[k].to(device), torch.full((n, 1), ts[k], device=device)) > -0.05).float().mean().item()
            for k in range(n_steps + 1)]
    top_sde = 100 * (sde[n_steps // 2][:, 1] > 0).float().mean().item()
    top_gen = 100 * (gen[n_steps // 2][:, 1] > 0).float().mean().item()
    m = dict(course=COURSE, seed=SEED, dim=DIM, symmetric=bool(C.get("symmetric", False)),
             reflect=bool(REFLECT), sde_contacts=contacts, sde_max_depth=deepest, sde_contacts_other=other_contacts,
             sde_qualified_T_other=other_qual,
             sde_illegal_max=max(ill_sde), gen_illegal_mean=float(np.mean(ill_gen)),
             gen_illegal_max=max(ill_gen), sde_out_of_bounds_max=max(oob_sde), gen_out_of_bounds_max=max(oob_gen),
             sde_qualified_T=qualified(sde[n_steps]), gen_qualified_T=qualified(gen[n_steps]),
             sde_top_half=top_sde, gen_top_half=top_gen,
             sde_symmetry_error=symmetry_error(list(sde)), gen_symmetry_error=symmetry_error(gen),
             sde_speed_p50=spd_sde.median().item(), sde_speed_p99=spd_sde.quantile(0.99).item(),
             gen_speed_p50=spd_gen.median().item(), gen_speed_p99=spd_gen.quantile(0.99).item(),
             sde_speed_change=jerk_sde, sde_near_wall=100 * float(np.mean(near)),
             sde_tunnel=tunnelling(list(sde)), gen_tunnel=tunnelling(gen))
    log(f"[{COURSE} seed {SEED}]  (SDE = agents following phi, GEN = generator density)", fname="metrics.txt", fresh=True)
    mode = "with reflection" if REFLECT else "NO reflection: only phi keeps them out"
    log(f"SDE agents, {mode}", fname="metrics.txt")
    log(f"inside obstacles:   SDE max {m['sde_illegal_max']:.2f}% at one time; {contacts:.1f}% of agents were inside one"
        f" at some moment, deepest {deepest:.3f}   |  GEN mean {m['gen_illegal_mean']:.2f}%, max {m['gen_illegal_max']:.2f}%",
        fname="metrics.txt")
    log(f"same agents {'without' if REFLECT else 'with'} reflection: {other_contacts:.1f}% "
        + ("were inside an obstacle at some moment" if REFLECT else "needed it")
        + f", {other_qual:.1f}% in the zone at T", fname="metrics.txt")
    log(f"paths through walls: SDE {m['sde_tunnel']:.2f}%   |  GEN {m['gen_tunnel']:.2f}%   (share of paths that ever cut"
        f" through a wall box between two frames)", fname="metrics.txt")
    log(f"out of the map:     SDE max {m['sde_out_of_bounds_max']:.2f}%   |  GEN max {m['gen_out_of_bounds_max']:.2f}%",
        fname="metrics.txt")
    log(f"qualified at T:     SDE {m['sde_qualified_T']:.1f}%   |  GEN {m['gen_qualified_T']:.1f}%", fname="metrics.txt")
    log(f"speed p50 / p99:    SDE {m['sde_speed_p50']:.1f} / {m['sde_speed_p99']:.1f}   |  GEN {m['gen_speed_p50']:.1f}"
        f" / {m['gen_speed_p99']:.1f}   (VMAX {VMAX:.1f}; SDE frame to frame change {jerk_sde:.2f})", fname="metrics.txt")
    log(f"hugging obstacles:  SDE {m['sde_near_wall']:.2f}% of agent time within 0.05 of an obstacle", fname="metrics.txt")
    log(f"top share at T/2:   SDE {top_sde:.1f}%   |  GEN {top_gen:.1f}%"
        + ("   (symmetric course: 50% expected)" if m["symmetric"] else "   (course not symmetric)"), fname="metrics.txt")
    log(f"symmetry error:     SDE {m['sde_symmetry_error']:.1f}   |  GEN {m['gen_symmetry_error']:.1f}   (average"
        f" |top share - 50| over the run, agents off the centre line; 0 = perfectly mirror symmetric)", fname="metrics.txt")
    with open(os.path.join(OUT_DIR, "metrics.csv"), "w", encoding="utf-8") as f:
        f.write("t,illegal_sde,illegal_generator,out_of_bounds_sde,out_of_bounds_generator\n")
        for k in range(n_steps + 1):
            f.write(f"{ts[k]:.4f},{ill_sde[k]:.3f},{ill_gen[k]:.3f},{oob_sde[k]:.3f},{oob_gen[k]:.3f}\n")
    fig.tight_layout(), save("snapshots.png", lambda p: fig.savefig(p, dpi=150)), plt.show()
    return m

def plot_paths(phi, n=150, steps=100):
    """Real agent routes: each agent follows the learned control v* = -grad_p H(grad phi) with the noise
    switched off (its plan). Unlike t maps to G(z, t), these are genuine trajectories."""
    z = sample_rho0(n)
    paths, cmap = simulate_sde(phi, z, steps, noise=False), plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_course(ax, C["draw_t"], course_image())
    for k in range(n):
        ax.plot(paths[:, k, 0], paths[:, k, 1], color=cmap((z[k, 1].item() + 2) / 4), lw=0.8, alpha=0.7, zorder=4)
    ax.set_title(f"agent routes under the learned control, noise off (hammers drawn at $t={C['draw_t']}$)")
    fig.tight_layout(), save("routes.png", lambda p: fig.savefig(p, dpi=150)), plt.show()

def make_gif(G, phi, n=600, frames=61, fname="fall_guys_apacnet.gif"):
    """Left: generator crowd. Right: SDE agents. Background: value function phi_omega(x, t)."""
    z = sample_rho0(n)
    col, sde, img = z[:, 1].cpu(), simulate_sde(phi, z, frames - 1), course_image()
    xs, ys = np.meshgrid(np.linspace(*XL, 90), np.linspace(*YL, 60))
    grid = torch.zeros(xs.size, DIM, device=device)
    grid[:, 0] = torch.tensor(xs.ravel(), dtype=torch.float32, device=device)
    grid[:, 1] = torch.tensor(ys.ravel(), dtype=torch.float32, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    def update(k):
        t = k / (frames - 1) * T
        with torch.no_grad():
            V = phi(grid, torch.full((grid.shape[0], 1), t, device=device)).reshape(xs.shape).cpu().numpy()
        V = np.minimum(V, np.quantile(V, 0.95))                # walls make phi spike; clip for contrast
        for ax, pts, name in zip(axes, [gen_at(G, z, t), sde[k]], ["generator $G_\\theta$", "SDE agents"]):
            ax.clear(), ax.contourf(xs, ys, V, levels=20, cmap="viridis", alpha=0.35, zorder=1)
            draw_course(ax, t, img), scatter_agents(ax, pts, col)
            ax.set_title(f"{name}   t = {t:.2f}   qualified {qualified(pts):.0f}%")

    anim = FuncAnimation(fig, update, frames=frames)
    save(fname, lambda p: anim.save(p, writer=PillowWriter(fps=10)))
    plt.close(fig)

def run_course():
    """Everything for the course/seed chosen by the COURSE and SEED environment variables: load the model
    if it exists (unless RETRAIN=1), otherwise train it; then write the logs and all plots."""
    print(f"===== course: {COURSE} | seed: {SEED} | device: {device} (output folder: {OUT_DIR}) =====")
    if os.path.exists(CKPT) and os.environ.get("RETRAIN", "0") != "1":   # replot instantly during the talk
        G, phi, ck = Generator().to(device), Phi().to(device), torch.load(CKPT, map_location=device)
        G.load_state_dict(ck["G"]), phi.load_state_dict(ck["phi"])
        hist, info = ck["hist"], ck.get("info", {})
    else:
        G, phi, hist = train()
        info = train.info
    write_csv(hist)
    plot_training(hist)
    m = plot_snapshots(G, phi)
    m["hjb_residual_end"] = float(np.mean(hist["HJB residual"][-2000:]))
    m["loss_phi_min"] = float(np.min(hist["loss_phi"]))
    m["device"], m["it_per_s"] = info.get("device", "?"), float(info.get("it_per_s", 0.0))
    # runaway (loss_phi diving far below its final level) or a residual that never came down
    m["training_ok"] = bool(m["loss_phi_min"] > float(np.mean(hist["loss_phi"][-2000:])) - 30 and m["hjb_residual_end"] < 1.5)
    log(f"training:           HJB residual at the end {m['hjb_residual_end']:.2f}, lowest loss_phi {m['loss_phi_min']:.1f}"
        f"  ->  {'OK' if m['training_ok'] else 'SUSPICIOUS (runaway or residual too high)'}", fname="metrics.txt")
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    plot_paths(phi), make_gif(G, phi)
    plt.close("all")
    return G, phi, hist

def write_summary(root="."):
    """Collect metrics.json of every out_* folder under root into summary.csv, summary.txt and summary.png."""
    rows = []
    for d in sorted(os.listdir(root)):
        f = os.path.join(root, d, "metrics.json")
        if d.startswith("out_") and os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                rows.append(json.load(fh))
    if not rows:
        return print("no metrics.json found")
    keys = list(rows[0])
    with open(os.path.join(root, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    cols = [("run", 24), ("SDE in obst", 11), ("SDE entered", 11), ("deepest", 7), ("SDE hug", 7), ("GEN in obst max", 15),
            ("GEN out map", 11), ("SDE qual", 8), ("GEN qual", 8), ("SDE sym err", 11), ("GEN sym err", 11),
            ("GEN tunnel", 10), ("SDE v99", 7), ("GEN v99", 7), ("HJB res", 7), ("train", 7), ("device", 6), ("it/s", 5)]
    lines = ["  ".join(n.rjust(w) for n, w in cols)]
    for r in rows:
        name = f"{r['course']}{' d' + str(r['dim']) if r.get('dim', 2) != 2 else ''} s{r['seed']}"
        sym = lambda v: f"{v:.1f}" if r["symmetric"] else "n/a"
        vals = [name, f"{r['sde_illegal_max']:.2f}", f"{r['sde_contacts']:.1f}", f"{r.get('sde_max_depth', 0):.3f}",
                f"{r.get('sde_near_wall', 0):.2f}",
                f"{r['gen_illegal_max']:.2f}", f"{r['gen_out_of_bounds_max']:.2f}", f"{r['sde_qualified_T']:.1f}",
                f"{r['gen_qualified_T']:.1f}", sym(r["sde_symmetry_error"]), sym(r["gen_symmetry_error"]),
                f"{r.get('gen_tunnel', 0):.2f}", f"{r.get('sde_speed_p99', 0):.1f}", f"{r.get('gen_speed_p99', 0):.1f}",
                f"{r['hjb_residual_end']:.2f}", "ok" if r.get("training_ok", True) else "SUSPECT",
                str(r.get("device", "?")), f"{r.get('it_per_s', 0):.1f}"]
        lines.append("  ".join(v.rjust(w) for v, (_, w) in zip(vals, cols)))
    lines.append("SDE agents " + ("with reflection" if rows[0].get("reflect") else "WITHOUT reflection: only phi keeps them"
                 " out of obstacles") + ". SDE in obst = largest % inside an obstacle at one time; SDE entered = % of"
                 " agents that were inside one at some moment; deepest = how far in the deepest one got.")
    lines.append("values in %, except v99 (99th percentile speed) and HJB res. sym err = average |top share - 50|"
                 " over the run, agents off the centre line (symmetric courses only)."
                 " hug = % of agent time within 0.05 of an obstacle. GEN tunnel = % of generator paths that ever"
                 " cut through a wall box.")
    with open(os.path.join(root, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    names = [f"{r['course']}\nseed {r['seed']}" for r in rows]
    panels = [("inside obstacles, max %", "sde_illegal_max", "gen_illegal_max"),
              ("qualified at T, %", "sde_qualified_T", "gen_qualified_T"),
              ("symmetry error, average over the run (symmetric courses)", "sde_symmetry_error", "gen_symmetry_error"),
              ("out of the map, max %", "sde_out_of_bounds_max", "gen_out_of_bounds_max"),
              ("99th percentile speed (VMAX about 12)", "sde_speed_p99", "gen_speed_p99")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(8, 1.3 * len(rows)), 3.0 * len(panels)))
    xs = np.arange(len(rows))
    for ax, (title, ks, kg) in zip(axes, panels):
        vs = [r.get(ks, 0) if (r["symmetric"] or "symmetry" not in ks) else 0 for r in rows]
        vg = [r.get(kg, 0) if (r["symmetric"] or "symmetry" not in kg) else 0 for r in rows]
        ax.bar(xs - 0.2, vs, 0.4, label="SDE agents"), ax.bar(xs + 0.2, vg, 0.4, label="generator")
        ax.set_title(title, fontsize=10), ax.set_xticks(xs), ax.set_xticklabels(names, fontsize=7)
    axes[0].legend(fontsize=8)
    fig.tight_layout(), fig.savefig(os.path.join(root, "summary.png"), dpi=150), plt.close(fig)
    print(f"saved {os.path.join(root, 'summary.png')}")

if __name__ == "__main__":
    run_course()
