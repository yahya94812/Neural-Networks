"""
One clean file, 12 sections, zero redundancy. Here's the flow logic so you know what to focus on as you read:

**Sections 1-2** are the "why" — numerical gradients show what a derivative *is* before autograd abstracts it away. `requires_grad=True` is the single switch that enables everything.

**Sections 3-4** are the most important conceptually — understanding that PyTorch builds a graph on every forward pass, and that gradients *accumulate* (so you must zero them). Most beginner bugs come from forgetting section 4.

**Sections 5-6** deepen the graph understanding — multiple variables, chain rule, and the subtle case where the same tensor appears in two branches (both paths contribute to `.grad`).

**Sections 7-8** are the transition from "understanding gradients" to "using them to optimize." Section 7 also introduces `torch.no_grad()` — which you need for updates, not just inference.

**Sections 9-12** build the full neural network stack. The progression is: one neuron → explicit `nn.Module` classes → `nn.Sequential` shorthand → full training loop with `torch.optim`.

The **Quick Reference** at the bottom is a cheat sheet you can glance at while writing your own code.
"""


# =============================================================================
# PyTorch Backpropagation — Complete Guide
# =============================================================================
# Covers (in order):
#   1. Numerical gradient (finite difference) — no PyTorch
#   2. torch.tensor and requires_grad
#   3. Autograd: how the computational graph is built and differentiated
#   4. Gradient accumulation and zeroing
#   5. Multi-variable graphs and chain rule
#   6. Same variable used multiple times (gradient routing)
#   7. Manual gradient descent with torch.no_grad()
#   8. Minimizing a function (converging example)
#   9. tanh activation and a single neuron
#  10. Building MLP from scratch with nn.Module
#  11. nn.Sequential shorthand
#  12. Full training loop with torch.optim
# =============================================================================

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# =============================================================================
# 1. NUMERICAL GRADIENT (Finite Difference)
# =============================================================================
# Before autograd existed, people computed gradients by hand using the
# limit definition:  df/dx ≈ (f(x+h) - f(x)) / h   for a very small h.
# This is slow and imprecise — autograd replaces it entirely — but it's
# useful to understand what a gradient *means* before PyTorch abstracts it.

def f_single(x):
    return 3*x**2 - 4*x + 5

h = 1e-6
x = 2 / 3
# Analytical derivative: df/dx = 6x - 4.  At x=2/3: 6*(2/3) - 4 = 0.0
numerical_grad = (f_single(x + h) - f_single(x)) / h
print(f"[1] Numerical df/dx at x=2/3 ≈ {numerical_grad:.6f}  (expected ~0.0)")


# Multi-variable: f(a, b, c) = a*b + c
def f_multi(a, b, c):
    return a * b + c

a, b, c = 2.0, -3.0, 10.0

# df/dc = 1.0 always (c appears with coefficient 1)
# df/da = b = -3.0,  df/db = a = 2.0
h = 1e-4
print(f"[1] Numerical df/da ≈ {(f_multi(a+h, b, c) - f_multi(a, b, c)) / h:.4f}  (expected -3.0)")
print(f"[1] Numerical df/db ≈ {(f_multi(a, b+h, c) - f_multi(a, b, c)) / h:.4f}  (expected  2.0)")
print(f"[1] Numerical df/dc ≈ {(f_multi(a, b, c+h) - f_multi(a, b, c)) / h:.4f}  (expected  1.0)")
print()


# =============================================================================
# 2. torch.tensor AND requires_grad
# =============================================================================
# A torch.tensor is PyTorch's fundamental data structure — it holds a
# multi-dimensional array (like numpy, but GPU-capable and differentiable).
#
# requires_grad=True is the key switch:
#   - It tells PyTorch: "track every operation on this tensor"
#   - PyTorch builds an internal computational graph as you do operations
#   - Calling .backward() on the final output then fills .grad on every
#     leaf tensor that had requires_grad=True
#
# Leaf tensor:  a tensor you created directly (like torch.tensor(...))
# Non-leaf:     result of an operation (like y = x * x)
#               PyTorch creates these automatically; you don't touch them.

x = torch.tensor(3.0, requires_grad=True)
# x.data  → the raw number inside (a 0-d tensor)
# x.grad  → gradient of the loss w.r.t. x (None until backward() is called)
# x.item() → convert 0-d tensor to a plain Python float

print(f"[2] x        = {x}")
print(f"[2] x.data   = {x.data}")
print(f"[2] x.grad   = {x.grad}  (None before backward)")
print()


# =============================================================================
# 3. AUTOGRAD: THE COMPUTATIONAL GRAPH
# =============================================================================
# When you write y = x * x with requires_grad=True on x:
#   - PyTorch records the multiplication in a graph node
#   - y stores a reference to x (its "parent") and the operation (**)
#   - y has a grad_fn (the backward function) but is NOT a leaf
#
# .backward() walks this graph from y → x, applying the chain rule at
# each node.  For y = x², the rule gives dy/dx = 2x.

x = torch.tensor(5.0, requires_grad=True)
y = x * x   # y = x²,  y.grad_fn = <MulBackward0>

print(f"[3] y           = {y.item()}")
print(f"[3] y.grad_fn   = {y.grad_fn}")   # shows the backward op
print(f"[3] y.is_leaf   = {y.is_leaf}")   # False — it was computed
print(f"[3] x.is_leaf   = {x.is_leaf}")   # True  — we created it

y.backward()   # dy/dx = 2x = 2*5 = 10

print(f"[3] x.grad after backward = {x.grad.item()}")   # 10.0
print()


# =============================================================================
# 4. GRADIENT ACCUMULATION AND ZEROING
# =============================================================================
# PyTorch ADDS to .grad on each backward call — it does NOT overwrite it.
# This is intentional (useful for gradient checkpointing and RNNs), but
# it means you MUST zero gradients before each new backward pass.
#
# Two equivalent ways to zero:
#   p.grad = None          ← cleanest; frees memory
#   p.grad.zero_()         ← keeps the tensor, fills with zeros (in-place)
#   optimizer.zero_grad()  ← calls zero_() on all params (covered in section 12)

x = torch.tensor(5.0, requires_grad=True)

# First backward: x.grad = 10
(x * x).backward()
print(f"[4] After 1st backward: x.grad = {x.grad.item()}")   # 10

# Second backward WITHOUT zeroing: x.grad = 10 + 10 = 20  (BUG!)
(x * x).backward()
print(f"[4] After 2nd backward (no zero): x.grad = {x.grad.item()}")  # 20 — WRONG

# Correct: zero first, then backward
x.grad = None
(x * x).backward()
print(f"[4] After zero + backward: x.grad = {x.grad.item()}")  # 10 — correct
print()


# =============================================================================
# 5. MULTI-VARIABLE GRAPH AND CHAIN RULE
# =============================================================================
# Every intermediate tensor is a node in the graph.
# backward() applies the chain rule at every node automatically.
#
# Example: L = (a*b + d) * fv
#
# Analytically:
#   dL/da  = b  * fv  = (-3)(-2) =  6
#   dL/db  = a  * fv  = (2)(-2)  = -4
#   dL/dd  = fv        = -2
#   dL/dfv = a*b + d   = (2*-3 + 10) = 4

a  = torch.tensor( 2.0, dtype=torch.float64, requires_grad=True)
b  = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
d  = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
fv = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)

# intermediate nodes — non-leaf, created automatically
c = a * b
e = c + d
L = e * fv

L.backward()

print(f"[5] dL/da  = {a.grad.item():.1f}   (expected  6.0)")
print(f"[5] dL/db  = {b.grad.item():.1f}  (expected -4.0)")
print(f"[5] dL/dd  = {d.grad.item():.1f}  (expected -2.0)")
print(f"[5] dL/dfv = {fv.grad.item():.1f}   (expected  4.0)")
print()


# =============================================================================
# 6. SAME VARIABLE USED MULTIPLE TIMES
# =============================================================================
# When a variable appears in multiple paths of the graph, PyTorch correctly
# sums the gradient contributions from ALL paths.
# This is because .grad uses += internally (same as the custom Value class).
#
# Example: f = (a*b) * (a+b)
#
#   df/da = d(a*b)/da*(a+b) + (a*b)*d(a+b)/da
#         = b*(a+b) + a*b
#         = b*(2a+b)  = 3*(2*(-2)+3) = 3*(-1) = -3
#
#   df/db = a*(a+b) + a*b
#         = a*(a+2b) = -2*(-2+6) = -8

a2 = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
b2 = torch.tensor( 3.0, dtype=torch.float64, requires_grad=True)

f2 = (a2 * b2) * (a2 + b2)   # a2 appears in TWO branches
f2.backward()

print(f"[6] df/da2 = {a2.grad.item():.1f}  (expected -3.0) — both paths summed")
print(f"[6] df/db2 = {b2.grad.item():.1f}  (expected -8.0)")
print()


# =============================================================================
# 7. MANUAL GRADIENT DESCENT WITH torch.no_grad()
# =============================================================================
# Now we use gradients to minimize a function iteratively.
#
# The update rule is:   param = param - lr * param.grad
#
# CRITICAL: the update itself must NOT be tracked by autograd.
# Wrapping it in torch.no_grad() tells PyTorch: "don't build a graph here."
# Without this, PyTorch would try to differentiate through the update,
# which is meaningless and creates graph nodes you don't want.
#
# Note: L = (a*b + d)*fv is linear in each variable, so there is no minimum
# — the parameters diverge toward -∞. The original notebook noted this too.
# The point here is just to see the mechanics of the update loop.

a  = torch.tensor( 2.0, dtype=torch.float64, requires_grad=True)
b  = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
d  = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
fv = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)

lr = 0.001

print("[7] Manual gradient descent on L = (a*b + d)*fv  (no minimum — diverges)")
for i in range(5):
    L = (a * b + d) * fv          # forward pass

    L.backward()                   # backward pass — fills .grad on a, b, d, fv

    with torch.no_grad():          # update — must be outside the graph
        a  -= lr * a.grad
        b  -= lr * b.grad
        d  -= lr * d.grad
        fv -= lr * fv.grad

    # Zero gradients AFTER the update, BEFORE the next forward pass
    a.grad  = None
    b.grad  = None
    d.grad  = None
    fv.grad = None

    print(f"    iteration {i}: L = {L.item():.4f}")
print()


# =============================================================================
# 8. MINIMIZING A FUNCTION (CONVERGING EXAMPLE)
# =============================================================================
# y = x²  has a global minimum at x=0.
# Gradient descent finds it: x ← x - lr * (2x)
# With lr=0.1:  x_new = x - 0.2x = 0.8x  → converges to 0.

x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
lr = 0.1

print("[8] Minimizing y = x²  (converges to x=0)")
for i in range(10):
    y = x * x                      # forward

    y.backward()                   # backward: x.grad = 2x

    with torch.no_grad():
        x -= lr * x.grad           # x ← x - 0.1 * 2x = 0.8x

    x.grad = None                  # zero grad

    print(f"    iteration {i}: y = {y.item():.6f},  x = {x.item():.6f}")
print()


# =============================================================================
# 9. TANH ACTIVATION AND A SINGLE NEURON
# =============================================================================
# A neuron computes:  output = tanh(x1*w1 + x2*w2 + b)
#
# tanh derivative:  d/dx tanh(x) = 1 - tanh²(x)
# So if o = tanh(n):
#   do/dw1 = x1 * (1 - o²)
#   do/dx1 = w1 * (1 - o²)
#
# With x1=2, w1=-3, o≈0.707:  1-o² ≈ 0.5
#   do/dw1 = 2  * 0.5 = 1.0
#   do/dx1 = -3 * 0.5 = -1.5

x1 = torch.tensor(2.0,                 dtype=torch.float64, requires_grad=True)
x2 = torch.tensor(0.0,                 dtype=torch.float64, requires_grad=True)
w1 = torch.tensor(-3.0,                dtype=torch.float64, requires_grad=True)
w2 = torch.tensor(1.0,                 dtype=torch.float64, requires_grad=True)
b  = torch.tensor(6.8813735870195432,  dtype=torch.float64, requires_grad=True)

n = x1*w1 + x2*w2 + b    # pre-activation (weighted sum)
o = torch.tanh(n)         # torch.tanh() — autograd differentiates through it

o.backward()

print(f"[9] o (tanh output) = {o.item():.6f}")
print(f"[9] do/dw1 = {w1.grad.item():.4f}  (expected  1.0000)")
print(f"[9] do/dx1 = {x1.grad.item():.4f}  (expected -1.5000)")
print(f"[9] do/dw2 = {w2.grad.item():.4f}  (expected  0.0000)  — x2 is 0")
print(f"[9] do/dx2 = {x2.grad.item():.4f}  (expected  0.5000)")
print()


# =============================================================================
# 10. BUILDING AN MLP WITH nn.Module
# =============================================================================
# nn.Module is PyTorch's base class for ALL neural network components.
# It mirrors the original Neuron → Layer → MLP hierarchy exactly.
#
# Key ideas:
#
#   nn.Parameter
#     A tensor that is automatically registered as a learnable parameter
#     when assigned as an attribute of an nn.Module. requires_grad=True
#     is set automatically. This replaces the Value objects in self.w / self.b.
#
#   nn.Linear(in_features, out_features)
#     Implements:  output = x @ weight.T + bias
#     - weight  shape: [out_features, in_features]  ← plays the role of self.w
#     - bias    shape: [out_features]                ← plays the role of self.b
#     Both are nn.Parameter, initialized with random values.
#
#   nn.ModuleList
#     A Python list that properly registers its contents as sub-modules.
#     Required so that model.parameters() can find all weights recursively.
#     A plain Python list would NOT work — parameters inside it would be invisible.
#
#   forward(x)
#     The computation to run when you call model(x).
#     PyTorch calls forward() inside __call__(), which also handles hooks, etc.
#     Always define forward(); never call it directly.
#
#   model.parameters()
#     Returns an iterator over all nn.Parameter objects in the module and
#     all its sub-modules recursively. This is what the optimizer uses.

class Neuron(nn.Module):
    """
    Single neuron: output = tanh(w·x + b)
    Directly mirrors the original:
        class Neuron:
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(random.uniform(-1,1))
    """
    def __init__(self, nin):
        super().__init__()              # always call super().__init__() first
        self.linear = nn.Linear(nin, 1) # packs self.w and self.b into one object

    def forward(self, x):
        act = self.linear(x)            # weighted sum + bias
        return torch.tanh(act).squeeze()# activation, squeeze to scalar


class Layer(nn.Module):
    """
    A layer of 'nout' neurons, each with 'nin' inputs.
    Mirrors:
        class Layer:
            self.neurons = [Neuron(nin) for _ in range(nout)]
    nn.Linear(nin, nout) is a vectorised batch of nout neurons in one matrix op.
    """
    def __init__(self, nin, nout):
        super().__init__()
        self.linear = nn.Linear(nin, nout)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class MLP(nn.Module):
    """
    Multi-layer perceptron.
    Mirrors:
        class MLP:
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    """
    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        # nn.ModuleList — the ONLY way to store a list of modules so that
        # .parameters() can find them. A plain list would hide them.
        self.layers = nn.ModuleList(
            [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    # No need to define parameters() — nn.Module handles it automatically


torch.manual_seed(42)
model = MLP(3, [4, 4, 1])   # 3 inputs → hidden(4) → hidden(4) → output(1)

print("[10] MLP architecture:")
print(model)
print(f"[10] Total learnable parameters: {sum(p.numel() for p in model.parameters())}")
print()


# =============================================================================
# 11. nn.Sequential — SHORTHAND FOR SIMPLE STACKS
# =============================================================================
# When the network is just a stack of layers with no branching logic,
# nn.Sequential is a cleaner alternative to writing a full nn.Module class.
# It calls each module in order during forward().
#
# Equivalent to MLP(3, [4, 4, 1]) above.

model_seq = nn.Sequential(
    nn.Linear(3, 4),  nn.Tanh(),   # Layer 1
    nn.Linear(4, 4),  nn.Tanh(),   # Layer 2
    nn.Linear(4, 1),  nn.Tanh(),   # Output
)

print("[11] Sequential equivalent:")
print(model_seq)
print(f"[11] Total parameters: {sum(p.numel() for p in model_seq.parameters())}")
print()


# =============================================================================
# 12. FULL TRAINING LOOP WITH torch.optim
# =============================================================================
# The manual loop in section 7 did:
#     with torch.no_grad():
#         for p in model.parameters():
#             p -= lr * p.grad
#     for p in model.parameters():
#         p.grad = None
#
# torch.optim encapsulates this cleanly:
#     optimizer.zero_grad()   ← zero all .grad
#     loss.backward()         ← fill .grad via backprop
#     optimizer.step()        ← p -= lr * p.grad  for every p
#
# SGD (Stochastic Gradient Descent) is the simplest optimizer — it's
# the direct equivalent of the manual update. Adam, AdamW, etc. add
# momentum and adaptive learning rates on top of the same idea.
#
# Training data: 4 samples, 3 features, targets in {-1, +1}
# Loss: Mean Squared Error  loss = mean((ypred - ytrue)²)

xs_data = [
    [ 2.0,  3.0, -1.0],
    [ 3.0, -1.0,  0.5],
    [ 0.5,  1.0,  1.0],
    [ 1.0,  1.0, -1.0],
]
ys_data = [1.0, -1.0, -1.0, 1.0]

X = torch.tensor(xs_data, dtype=torch.float32)         # shape [4, 3]
Y = torch.tensor(ys_data, dtype=torch.float32).unsqueeze(1)  # shape [4, 1]

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(3, 4), nn.Tanh(),
    nn.Linear(4, 4), nn.Tanh(),
    nn.Linear(4, 1), nn.Tanh(),
)

# SGD with lr=0.1 — same hyperparameter as the original notebook
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("[12] Training loop:")
for k in range(20):
    # ── Step 1: Forward pass ──────────────────────────────────────
    ypred = model(X)                           # shape [4, 1]
    loss  = ((ypred - Y) ** 2).mean()          # MSE

    # ── Step 2: Zero gradients ────────────────────────────────────
    # Must come BEFORE backward, not after.
    # (Putting it after works too, but before is the standard convention.)
    optimizer.zero_grad()

    # ── Step 3: Backward pass ─────────────────────────────────────
    loss.backward()   # fills .grad on all parameters

    # ── Step 4: Update parameters ─────────────────────────────────
    optimizer.step()  # p.data -= lr * p.grad  for every p in model

    print(f"    k={k:2d}  loss = {loss.item():.6f}")

print()

# ── Inference: always use torch.no_grad() ─────────────────────────
# During inference you don't need the graph (no backward pass).
# torch.no_grad() skips building it → saves memory and speeds things up.
with torch.no_grad():
    preds = model(X)

print("[12] Final predictions vs targets:")
for pred, target in zip(preds.squeeze().tolist(), ys_data):
    sign = "✓" if (pred > 0) == (target > 0) else "✗"
    print(f"    pred: {pred:+.4f}   target: {target:+.1f}   {sign}")


# =============================================================================
# QUICK REFERENCE
# =============================================================================
# torch.tensor(x, requires_grad=True)  — create a differentiable scalar/tensor
# v.item()                             — extract Python float from 0-d tensor
# v.grad                               — gradient (None until backward() called)
# loss.backward()                      — run backprop, fill .grad on all leaves
# p.grad = None                        — zero gradients (do this every iteration)
# torch.no_grad()                      — context manager: no graph, no grads
# nn.Module                            — base class for all network components
# nn.Linear(in, out)                   — fully-connected layer (weight + bias)
# nn.Tanh() / torch.tanh()             — tanh activation
# nn.ModuleList([...])                 — list of modules that .parameters() sees
# nn.Sequential(...)                   — ordered stack of modules
# model.parameters()                   — iterator over all learnable parameters
# optimizer.zero_grad()                — zero all .grad before backward
# optimizer.step()                     — apply gradient update to all parameters
# with torch.no_grad():                — disable autograd for inference / updates
