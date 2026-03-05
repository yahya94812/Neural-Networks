# Components Of CUDA

Alright — let’s build the **full mental model of CUDA**, from hardware all the way up to your code. Once this clicks, GPU programming suddenly makes a lot more sense.

---

# 🧱 The CUDA Stack (top → bottom)

Think of CUDA as layers:

```
Your Application (PyTorch / C++ / Python)
        ↓
CUDA Libraries (cuBLAS, cuDNN, etc.)
        ↓
CUDA Runtime API
        ↓
CUDA Driver
        ↓
GPU Hardware (SMs, cores, memory)
```

Let’s walk through each piece.

---

## 1️⃣ Application Layer (what you write)

This is your code using:

* C/C++ CUDA kernels
* Python with PyTorch / TensorFlow
* Numba CUDA
* CUDA Fortran

Example:

```cpp
vectorAdd<<<blocks, threads>>>(A, B, C);
```

At this level you think in terms of:

* threads
* blocks
* grids

---

## 2️⃣ CUDA Libraries (high-performance building blocks)

Provided by NVIDIA.

Common ones:

* **cuBLAS** → linear algebra
* **cuDNN** → deep learning
* **cuFFT** → fast Fourier transform
* **Thrust** → STL-like GPU library

👉 Most ML engineers live here — they rarely write kernels.

---

## 3️⃣ CUDA Runtime API

This is the friendly layer that:

* launches kernels
* manages memory
* handles device selection

Examples:

```cpp
cudaMalloc()
cudaMemcpy()
kernel<<<...>>>()
```

Think of it as the **convenience wrapper**.

---

## 4️⃣ CUDA Driver

Lower-level component that:

* talks directly to GPU
* loads kernels
* manages contexts

Usually you don’t touch this unless doing advanced work.

---

## 5️⃣ GPU Hardware (where the magic happens)

Now we hit the most important part.

---

# 🧠 CUDA Hardware Architecture

Modern NVIDIA GPUs are organized like this:

```
GPU
 ├── GPCs (Graphics Processing Clusters)
 │     ├── SM (Streaming Multiprocessor)
 │     │     ├── CUDA cores
 │     │     ├── Tensor cores (optional)
 │     │     ├── Shared memory
 │     │     └── Warp schedulers
```

Let’s decode.

---

## 🔹 Streaming Multiprocessor (SM)

**SM = the real workhorse**

Each SM contains:

* many CUDA cores
* registers
* shared memory
* warp scheduler

👉 Blocks are assigned to SMs.

**Key idea:**
SM executes many threads concurrently.

---

## 🔹 CUDA Cores

These are the actual arithmetic units.

They perform:

* add
* multiply
* FMA
* logic ops

Think of them as **tiny ALUs**.

Example GPU:

* RTX 4090 → ~16,000 CUDA cores
* But only dozens of SMs

Because:

👉 Threads are time-sliced on cores.

---

## 🔹 Tensor Cores (specialized units)

Found in modern GPUs.

Used for:

* matrix multiply
* deep learning
* mixed precision

Much faster for AI workloads.

---

# 🧵 CUDA Execution Model (software view)

Now the part everyone struggles with: **threads, blocks, grids**.

---

## 🌐 Grid → Block → Thread hierarchy

```
Grid
 ├── Block 0
 │     ├── Thread 0
 │     ├── Thread 1
 │     └── ...
 ├── Block 1
 └── ...
```

This is purely a **software abstraction**.

---

# 🧵 What is a Thread?

### Definition

A **thread** is the smallest execution unit in CUDA.

Each thread:

* runs the kernel code
* has its own registers
* has its own thread ID
* executes the same program

👉 CUDA uses **SIMT** (Single Instruction, Multiple Threads).

---

### Example mental model

Vector addition with N = 1,000,000:

* 1 thread handles element 0
* another handles element 1
* etc.

Thousands run in parallel.

---

### Thread indexing

Inside kernel:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

This gives each thread a unique global ID.

---

# 🧱 What is a Block?

### Definition

A **block** is a group of threads that:

✅ run on the SAME SM
✅ can synchronize
✅ can share fast shared memory

---

### Why blocks exist

Hardware constraint:

👉 Threads that need to cooperate must be scheduled together.

So CUDA groups them into blocks.

---

### Important block properties

Threads in a block can:

* use `__syncthreads()`
* share **shared memory**
* communicate efficiently

Threads in different blocks:

❌ cannot synchronize
❌ cannot share shared memory

---

# 🌐 What is a Grid?

### Definition

A **grid** is the collection of all blocks launched for a kernel.

When you launch:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>();
```

You are creating:

* one grid
* many blocks
* many threads

---

# ⚙️ How CUDA maps software → hardware

This is the KEY insight.

---

## Step-by-step execution

### 1️⃣ You launch kernel

```cpp
kernel<<<grid, block>>>();
```

---

### 2️⃣ CUDA scheduler assigns blocks to SMs

Important:

* Blocks are scheduled dynamically
* Order is NOT guaranteed
* Multiple blocks can run on one SM

---

### 3️⃣ SM executes threads in warps

---

# 🌀 What is a Warp? (VERY important)

A **warp** is:

> A group of **32 threads** executed together in lockstep.

This is the true hardware execution unit.

---

## Warp facts

* Size = 32 threads
* Executes one instruction at a time
* Managed by warp scheduler
* Hidden from most programmers

---

## Why warps matter

If threads in a warp diverge:

```cpp
if (threadIdx.x % 2 == 0)
```

GPU must serialize execution → slower.

This is called **warp divergence**.

---

# 🧠 Memory hierarchy (quick but crucial)

CUDA memory speed (fast → slow):

1. Registers (per thread)
2. Shared memory (per block)
3. L1 cache
4. Global memory (device DRAM)
5. Host memory

---

## Memory ownership

| Memory        | Scope  |
| ------------- | ------ |
| Registers     | thread |
| Shared memory | block  |
| Global memory | grid   |

---

# 🔥 Putting it all together (full flow)

When you run a CUDA program:

1. CPU launches kernel
2. Grid is created
3. Blocks scheduled to SMs
4. SM splits threads into warps
5. CUDA cores execute instructions
6. Results written to global memory

---

# 🧭 Mental cheat sheet

**Software hierarchy**

```
Grid > Block > Thread
```

**Hardware hierarchy**

```
GPU > SM > Warp > CUDA core
```

**Mapping**

* Grid → entire GPU
* Block → SM
* Warp → execution group
* Thread → logical worker

---

# ✅ When you truly understand CUDA

You can answer:

* How many threads per block?
* How many blocks per SM?
* Is memory coalesced?
* Is warp divergence happening?

That’s when performance tuning begins.

---