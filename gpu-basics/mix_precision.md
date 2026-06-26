- Deep learning models are highly tolerant of slight precision losses during multiplication, but they are extremely sensitive to precision losses during summation (accumulation). If you add thousands of tiny numbers together in low precision, rounding errors quickly accumulate and destroy the model's accuracy.

- MMA : matrix multiply-accumulate (tensor cores are specialized in it)
- FFMA : fused floating point multiply add (std operation in cuda cores and in other processors (not tensor cores)) eg. D = (A * B) + C in a single round

- You cant use A100-tensor-core-MMA-supported-dtypes from torch.amp.autocast() because it allow only the following input and accumulator dtypes
    - float16 and bfloat16 but in both case the accumulator is FP32 dtype
    - FP16 Input / FP32 Accumulator: When you run autocast with dtype=torch.float16, PyTorch automatically targets this mode. For example, the inputs to a Linear layer or Convolution are cast down to 16-bit to hit the high TOPS matrix cores, but critical accumulation steps (like reductions and loss calculations) automatically retain FP32 precision for numerical safety.
    - BF16 Input / FP32 Accumulator: Using dtype=torch.bfloat16 triggers this exact hardware pipeline.

- input and output is FP32 but internally mat mul happened at FP16 or BF16 or TF32 

- torch.set_float32_matmul_precision('high') # it uses TF32 during matmul; if set to highest it will use FP32
- not it improve the compute speed not the memory overhead because it still memory overyhead

- by default all the weight and activation and all the tensors are using FP32
- int8 is for inference
- deep learning is memory bound

Whether you type `torch.autocast` or `torch.amp.autocast`, you are calling the exact same underlying context manager (`torch.amp.autocast_mode.autocast`) under the hood.

- **Multiplication ($A \times B$):** The inputs $A$ and $B$ (your weights and activations) have been cast to **FP16** by autocast. The hardware multiplies them together very quickly in FP16.
    
- **Accumulation ($+ C$):** Even though the inputs were FP16, the Tensor Core performs the intermediate addition (the accumulation) in **FP32**. This is a hardware-level safety net designed to prevent the numerical underflow/overflow that would happen if you tried to sum thousands of tiny FP16 numbers together.
    
- **Output ($D$):** The final output block is yielded back. Depending on the next operation in the graph, `autocast` either leaves it as FP16 or casts it back to FP32 for safer downstream processing (like activations or layer norms).

# the accumlator is the last additon term C and D (both are same)
eg in assembly
ADD B; B + A -> A 
like this 