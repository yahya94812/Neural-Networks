# GPU vs Accelerator Card

1. A **GPU** is the compute chip (die), while the rectangular device with fans gamers buy is a **graphics (accelerator) card** that contains the GPU.
2. **NVIDIA** mainly designs GPU chips and also sells some complete accelerator cards (Founders Edition), while most retail cards are built by partners.
3. For neural network training, buy a discrete NVIDIA accelerator card with sufficient VRAM (e.g., **GeForce RTX 4090** for heavy workloads).
4. The RTX 4090 accelerator card contains the **GA102** GPU chip.
5. The **NVIDIA A100** is a data-center accelerator product that contains the GA100 GPU chip and is not used in consumer graphics cards.
6. In practice, “A100” refers to the full accelerator module/card, while **GA100** is the underlying GPU silicon.
7. The **NVIDIA Tesla P100** is likewise a complete accelerator product containing the GP100 GPU chip.
8. CUDA compatibility is determined by the GPU architecture (compute capability), not by the specific board vendor’s accelerator card.
9. VRAM capacity is mostly fixed by NVIDIA’s product SKU; board vendors usually cannot change it beyond allowed configurations.
10. GPUs may use GDDR (common graphics VRAM), HBM (high-bandwidth stacked memory), on-chip SRAM caches, and host system RAM via PCIe/NVLink.

# GPU Architecture & Components

## Hierarchical Structure

A GPU is built around a large silicon die (e.g., NVIDIA's GA102). The majority of the die's area is occupied by processing cores, organized in a strict hierarchy:

**Die → GPC → SM → Warp → CUDA Cores**

| Level | Unit | Count |
|---|---|---|
| Die | Graphics Processing Cluster (GPC) | 7 |
| GPC | Streaming Multiprocessor (SM) | 12 per GPC |
| SM | Warps + 1 Ray Tracing (RT) Core | 4 warps per SM |
| Warp | CUDA/Shading Cores + 1 Tensor Core | 32 CUDA cores per warp |

So at full scale: 7 GPCs × 12 SMs × 4 warps × 32 CUDA cores = **10,752 CUDA cores** total.

## Parallel Execution Model

**Embarrassingly Parallel** problems are a class of computational problems that require little to no effort to divide into parallel tasks — each unit of work is fully independent (e.g., rendering pixels, matrix multiplication). GPUs are purpose-built for this type of workload.

## SIMD — Single Instruction, Multiple Data

- Each instruction corresponds to a **thread**, and each thread maps to a **CUDA core**
- Threads are grouped into **warps** (32 threads each), and the same instruction is issued simultaneously to all threads in a warp
- Warps are bundled into **thread blocks**, each of which is assigned to and managed by a single SM
- Thread blocks are further grouped into a **grid**, which represents the full computation distributed across the entire GPU
- All threads within a warp execute in lock-step — like soldiers marching in unison — meaning they must all execute the same instruction at the same time

## SIMT — Single Instruction, Multiple Threads

SIMT is NVIDIA's extension of the SIMD model that adds per-thread flexibility:

- Each thread has its own **program counter** and **register state**, allowing threads to diverge in execution path
- Threads can execute at different speeds and take different branches
- Threads within a warp can share memory (shared memory / L1 cache within the SM)
- This flexibility enables **warp divergence** handling — when threads in a warp hit a conditional branch (e.g., `if/else`), the GPU serializes the divergent paths, executing each branch for the relevant threads while masking the others, then reconverges afterward
- SIMT provides the programming flexibility of individual threads while retaining the hardware efficiency of SIMD execution