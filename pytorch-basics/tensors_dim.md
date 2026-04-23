
# 🧠 PyTorch Matrix Dimensions — Notes

---

## 1️⃣ Tensor Shapes

Every PyTorch tensor has a shape:

```python
x = torch.randn(3, 4)
print(x.shape)  # torch.Size([3, 4])
```

**Interpretation**

* First number → rows
* Second number → columns

**General rule**

```
shape = (dim0, dim1, dim2, …)
```

---

## 2️⃣ Matrix Multiplication (`@` / `torch.matmul`)

### Core Rule

```
(A × B) @ (B × C) → (A × C)
```

**Requirements**

* Inner dimensions must match
* Otherwise PyTorch raises an error

---

### Example

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

C = A @ B
print(C.shape)  # (2, 4)
```

---

### Common Error

```python
A = torch.randn(2, 3)
B = torch.randn(2, 4)

A @ B  # ❌ invalid (3 ≠ 2)
```

---

## 3️⃣ Batched Matrix Multiplication

PyTorch automatically handles batches.

### Rule

```
(batch, A, B) @ (batch, B, C)
        ↓
(batch, A, C)
```

---

### Example

```python
A = torch.randn(5, 2, 3)
B = torch.randn(5, 3, 4)

C = torch.matmul(A, B)
print(C.shape)  # (5, 2, 4)
```

**Key idea**

👉 Matrices are multiplied **pairwise across the batch dimension**

---

## 4️⃣ Dimension Manipulation

### ➕ `unsqueeze`

Adds a dimension of size 1.

```python
x = torch.randn(4)      # (4,)
x = x.unsqueeze(0)      # (1, 4)
```

**Common use**

* Adding batch dimension
* Preparing inputs for models

---

### ➖ `squeeze`

Removes dimensions of size 1.

```python
x = torch.randn(1, 4)
x = x.squeeze(0)        # (4,)
```

---

## 5️⃣ Reshape vs View

### ✅ `reshape()` (recommended)

* Safe
* Works on non-contiguous tensors

```python
y = x.reshape(new_shape)
```

---

### ⚡ `view()`

* Faster
* Requires contiguous memory

```python
y = x.view(new_shape)
```

**Beginner rule**

> Prefer `reshape()` unless you know why you need `view()`.

---

## 6️⃣ Transpose vs Permute

### 🔁 `transpose(dim1, dim2)`

Swaps two dimensions.

```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)  # (3, 2)
```

---

### 🔀 `permute(...)`

Reorders multiple dimensions.

```python
x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1)  # (4, 2, 3)
```

**Typical use**

* Images
* Transformers
* Complex tensor layouts

---

## 7️⃣ Attention Shape Flow

Given:

```python
k = self.key(x)   # (B, T, hs)
q = self.query(x) # (B, T, hs)
wei = q @ k.transpose(-2, -1)
```

### Goal

Compute attention scores.

---

### Required shapes

```
q: (B, T, hs)
kᵀ: (B, hs, T)
-----------------
wei: (B, T, T)
```

---

### Why transpose is needed

To satisfy matmul rule:

```
(T × hs) @ (hs × T)
```

---

## 8️⃣ Negative Indexing (Important)

Negative indices count from the end.

For shape `(B, T, hs)`:

| Index | Meaning |
| ----- | ------- |
| -1    | hs      |
| -2    | T       |
| -3    | B       |

---

### Recommended pattern

```python
k.transpose(-2, -1)
```

**Meaning**

👉 Swap the last two dimensions

---

## 9️⃣ Why NOT use `transpose(1, 2)`?

### It works only when shape is fixed

For `(B, T, hs)`:

```python
k.transpose(1, 2)  # works
```

But breaks when tensor becomes:

```
(B, nh, T, hs)
```

---

### Why `transpose(-2, -1)` is better

✅ Works for any tensor rank
✅ Robust for multi-head attention
✅ Standard PyTorch practice
✅ Future-proof

---

## 🔟 Debugging Dimension Errors

### Best practices

✅ Always print shapes

```python
print(x.shape)
```

✅ Before matmul, check:

* shape of A
* shape of B
* inner dimensions match

---

### Powerful mental trick

When stuck, write:

```
(?, ?) @ (?, ?)
```

and fill in numbers.

---

# 🚀 Quick Summary

You now understand:

* Tensor shapes
* Matrix multiplication rules
* Batched matmul
* Unsqueeze / squeeze
* Reshape vs view
* Transpose vs permute
* Attention dimension flow
* Negative indexing best practices
* Debugging workflow

---