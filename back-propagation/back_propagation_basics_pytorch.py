import torch
import torch.nn as nn

# =============================================================================
# 1. NUMERICAL GRADIENT (sanity check — autograd replaces this)
# =============================================================================
f = lambda x: 3*x**2 - 4*x + 5
h = 1e-6; x = 2/3
print(f"[1] df/dx ≈ {(f(x+h) - f(x)) / h:.6f}  (expected 0.0)")

# =============================================================================
# 2. requires_grad — the single switch that enables autograd
# =============================================================================
# x.grad  → filled after .backward()
# x.item() → tensor → Python float
x = torch.tensor(5.0, requires_grad=True)   # leaf tensor

# =============================================================================
# 3. COMPUTATIONAL GRAPH + BACKWARD
# =============================================================================
y = x * x          # y.grad_fn is recorded; y is NOT a leaf
y.backward()       # dy/dx = 2x = 10
print(f"[3] x.grad = {x.grad.item()}")  # 10.0

# =============================================================================
# 4. GRADIENT ACCUMULATION — zero before every backward
# =============================================================================
# PyTorch *adds* to .grad; it never resets it automatically.
x = torch.tensor(5.0, requires_grad=True)
(x*x).backward(); print(f"[4] grad after 1st: {x.grad.item()}")  # 10
(x*x).backward(); print(f"[4] grad after 2nd (bug!): {x.grad.item()}")  # 20 — wrong

x.grad = None      # or x.grad.zero_()  or optimizer.zero_grad()
(x*x).backward(); print(f"[4] grad after zero+backward: {x.grad.item()}")  # 10

# =============================================================================
# 5. MULTI-VARIABLE GRAPH  —  L = (a*b + d) * fv
# =============================================================================
# dL/da=b*fv=6, dL/db=a*fv=-4, dL/dd=fv=-2, dL/dfv=a*b+d=4
a  = torch.tensor( 2.0, dtype=torch.float64, requires_grad=True)
b  = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
d  = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
fv = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
L  = (a*b + d) * fv
L.backward()
print(f"[5] dL/da={a.grad:.1f}  dL/db={b.grad:.1f}  dL/dd={d.grad:.1f}  dL/dfv={fv.grad:.1f}")

# =============================================================================
# 6. SAME VARIABLE IN MULTIPLE BRANCHES — gradients are summed
# =============================================================================
# f = (a*b)*(a+b)  →  df/da = b*(2a+b) = -3, df/db = a*(a+2b) = -8
a2 = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
b2 = torch.tensor( 3.0, dtype=torch.float64, requires_grad=True)
((a2*b2) * (a2+b2)).backward()
print(f"[6] df/da2={a2.grad:.1f}  df/db2={b2.grad:.1f}")

# =============================================================================
# 7-8. GRADIENT DESCENT — wrap updates in torch.no_grad()
# =============================================================================
# torch.no_grad() prevents the update from being added to the graph.
x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
for i in range(10):
    y = x*x
    y.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad   # x ← 0.8x → converges to 0
    x.grad = None
    print(f"  [8] iter {i}: x={x.item():.4f}")

# =============================================================================
# 9. SINGLE NEURON  —  o = tanh(x1*w1 + x2*w2 + b)
# =============================================================================
x1 = torch.tensor(2.0,                dtype=torch.float64, requires_grad=True)
x2 = torch.tensor(0.0,                dtype=torch.float64, requires_grad=True)
w1 = torch.tensor(-3.0,               dtype=torch.float64, requires_grad=True)
w2 = torch.tensor(1.0,                dtype=torch.float64, requires_grad=True)
b  = torch.tensor(6.8813735870195432, dtype=torch.float64, requires_grad=True)
o  = torch.tanh(x1*w1 + x2*w2 + b)
o.backward()
print(f"[9] o={o.item():.4f}  do/dw1={w1.grad:.4f}  do/dx1={x1.grad:.4f}")

# =============================================================================
# 10. nn.Module — explicit class hierarchy
# =============================================================================
# nn.Linear(nin, nout)  →  output = x @ W.T + b  (W, b are nn.Parameter)
# nn.ModuleList         →  the ONLY way to store module lists so
#                          .parameters() can find them (plain list won't work)
# Always call super().__init__() and define forward(); never call it directly.

class MLP(nn.Module):
    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        self.layers = nn.ModuleList(
            [nn.Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
        )
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.tanh(layer(x)) if i < len(self.layers)-1 else layer(x)
        return x

# =============================================================================
# 11. nn.Sequential — shorthand for straight stacks (no custom logic needed)
# =============================================================================
model = nn.Sequential(
    nn.Linear(3, 4), nn.Tanh(),
    nn.Linear(4, 4), nn.Tanh(),
    nn.Linear(4, 1), nn.Tanh(),
)
print(f"[11] params: {sum(p.numel() for p in model.parameters())}")

# =============================================================================
# 12. FULL TRAINING LOOP
# =============================================================================
# optimizer.zero_grad()  ←  zero all .grad
# loss.backward()        ←  fill .grad via backprop
# optimizer.step()       ←  p -= lr * p.grad  for every p

X = torch.tensor([[ 2., 3.,-1.],[ 3.,-1., .5],[ .5, 1., 1.],[ 1., 1.,-1.]])
Y = torch.tensor([[1.],[-1.],[-1.],[1.]])

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(3,4), nn.Tanh(),
    nn.Linear(4,4), nn.Tanh(),
    nn.Linear(4,1), nn.Tanh(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for k in range(20):
    loss = ((model(X) - Y)**2).mean()  # MSE forward
    optimizer.zero_grad()              # zero grads
    loss.backward()                    # backward
    optimizer.step()                   # update
    print(f"  [12] k={k:2d}  loss={loss.item():.6f}")

# Inference: disable graph building — saves memory, speeds up
with torch.no_grad():
    preds = model(X).squeeze().tolist()
for p, t in zip(preds, [1.,-1.,-1.,1.]):
    print(f"  pred={p:+.3f}  target={t:+.1f}  {'✓' if (p>0)==(t>0) else '✗'}")

# =============================================================================
# QUICK REFERENCE
# =============================================================================
# torch.tensor(x, requires_grad=True)  create differentiable tensor
# loss.backward()                       backprop → fills .grad on all leaves
# p.grad = None / p.grad.zero_()        zero grads before next backward
# torch.no_grad()                       no graph (use for updates & inference)
# nn.Linear(in, out)                    weight + bias, both nn.Parameter
# nn.ModuleList([...])                  list visible to .parameters()
# nn.Sequential(...)                    ordered stack, auto-forward
# optimizer.zero_grad()                 zero all grads
# optimizer.step()                      apply p -= lr * p.grad to all params