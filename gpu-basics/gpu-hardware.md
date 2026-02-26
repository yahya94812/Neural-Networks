**Precise Notes — GPU vs Accelerator Card**

1. A **GPU** is the compute chip (die), while the rectangular device with fans gamers buy is a **graphics (accelerator) card** that contains the GPU.
2. **NVIDIA** mainly designs GPU chips and also sells some complete accelerator cards (Founders Edition), while most retail cards are built by partners.
3. For neural network training, buy a discrete NVIDIA accelerator card with sufficient VRAM (e.g., **GeForce RTX 4090** for heavy workloads).
4. The RTX 4090 accelerator card contains the **AD102** GPU chip.
5. The **NVIDIA A100** is a data-center accelerator product that contains the GA100 GPU chip and is not used in consumer graphics cards.
6. In practice, “A100” refers to the full accelerator module/card, while **GA100** is the underlying GPU silicon.
7. The **NVIDIA Tesla P100** is likewise a complete accelerator product containing the GP100 GPU chip.
8. CUDA compatibility is determined by the GPU architecture (compute capability), not by the specific board vendor’s accelerator card.
9. VRAM capacity is mostly fixed by NVIDIA’s product SKU; board vendors usually cannot change it beyond allowed configurations.
10. GPUs may use GDDR (common graphics VRAM), HBM (high-bandwidth stacked memory), on-chip SRAM caches, and host system RAM via PCIe/NVLink.