## Different types Core Compute Units in nvidia GPUs
1. CUDA Cores (always FP32/INT32 ALUs) (general-purpose)
    * General-purpose parallel math
    * Execute FP32, FP16, and INT32 instructions.
    * Organized into warps (32 threads) and SMs (Streaming Multiprocessors).
    * Think of them as the GPU’s basic workers.

2. Tensor Cores
    * These are purpose-built for matrix math.
    * Perform fused matrix multiply-accumulate (MMA)
    * Much higher throughput for: FP16, BF16, TF32, INT8, FP8(newer GPUs)

3. RT Cores (Ray Tracing Cores)
    * Hardware ray traversal & intersection
    * real-time ray tracing, path tracing
    * Offload heavy ray tracing math from CUDA cores.
    * Mostly graphics-focused but increasingly used in simulation.

## Mixed precision
* it is the method in which the operands are in small sizes say FP8 but the results of the operations are stored in higher size say FP16 (because it may leads to underflow and rounding errors)
* FP8 × FP8 → accumulate in FP16/FP32

## Measuring compute
### FLOPs = Floating Point Operations per Second
| Unit   | Meaning         |
| ------ | --------------- |
| FLOP   | 1 operation/sec |
| GFLOPs | 10⁹ ops/sec     |
| TFLOPs | 10¹² ops/sec    |
| PFLOPs | 10¹⁵ ops/sec    |
* FLOPs are always tied to:
    * precision (FP32 vs FP16 vs FP8)
    * hardware unit (CUDA vs Tensor)

### Floating-Point Precisions (FP)
* FP64 (Double Precision)
* FP32 (Single Precision)
* FP16 (Half Precision)
* BF16 (Brain Floating Point)
    * 16-bit
    * Same exponent range as FP32
    * Safer numerically than FP16
    * Popular in modern AI training
    * Used heavily on:
    * Ampere, Hopper
* FP8 (New hotness)
    * 8-bit floating point
    * Very low precision
* INT4 / INT1
* INT8

## connections
* PCIe(Peripheral Component Interconnect Express) = the standard cable that connects a GPU to the motherboard/CPU.
* NVLink = a much faster direct bridge between GPUs.
* Why it matters:
*     Single GPU → PCIe is enough
*     Multiple GPUs training together → NVLink is much faster
* Think: PCIe = normal road, NVLink = express highway between GPUs.

## Overall
### 🟩 CUDA Cores
The classic GPU cores.
Good at:
* FP32
* FP64 (sometimes)
* general compute
* graphics
But…
👉 They are **much slower than Tensor Cores for AI math**

* * *

### 🟪 Tensor Cores
Specialized matrix-multiply engines for AI.
They accelerate:
* FP16
* BF16
* FP8
* INT8
* sometimes FP32 (TensorFloat-32)
These are why modern NVIDIA GPUs quote huge AI TFLOPs numbers.

* * *

### Example
An H100 might have:
* FP32 CUDA: ~60 TFLOPs
* FP8 Tensor: **~2000+ TFLOPs**
Same chip, wildly different numbers.

* * *

### 📐 TensorFloat-32 (TF32) — NVIDIA’s special format
Introduced in:
* NVIDIA Ampere architecture
TF32 is:
* 19-bit effective precision
* runs on Tensor Cores
* keeps FP32 range
* faster than FP32 cuda
Used mainly for:
* AI training acceleration
* drop-in replacement for FP32 in many workloads

* * *

### 🚀 How to Read NVIDIA Spec Sheets (Practical)
When you see something like:
> 989 TFLOPs FP8 Tensor
Translate it as:
✅ theoretical peak  
✅ AI-specific  
✅ requires Tensor Core usage  
❌ not general compute speed

* * *

### 🧠 Rule of Thumb (very useful)
For modern NVIDIA GPUs:
* Graphics → look at FP32 CUDA
* AI training → look at FP16 / BF16 Tensor
* Cutting-edge AI → look at FP8 Tensor
* Inference → look at INT8 / FP8

* * *