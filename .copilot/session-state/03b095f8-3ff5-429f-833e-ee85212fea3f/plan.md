# Plan: Nemotron/NemotronH Model Support

## Completed
- [x] Nemotron + NemotronH model support (PR #23)
- [x] SSM precision fixes (CastLike, fp32 upcast)
- [x] GatedRMSNorm grouping + ORT CUDA workaround
- [x] Mamba2Block chunked SSD multi-token path
- [x] Mamba2Block Scan multi-token path
- [x] dt clamp (time_step_min) support
- [x] HF weight corruption root cause (rescale_prenorm_residual fix)
- [x] **Mamba2Block refactoring (Option B+D)** — split into base + 3 subclasses
  - `_mamba_block.py`: Mamba2BlockBase, Mamba2BlockSingle, factory Mamba2Block
  - `_mamba_block_scan.py`: Mamba2BlockScan
  - `_mamba_block_chunked.py`: Mamba2BlockChunkedSSD
  - All 2232 tests pass (2 pre-existing mamba2 failures unrelated)

## Branch
`rama/chunkscan` at `704a06c`

## Remaining
- [ ] Numerical parity investigation (layer-by-layer comparison)
- [ ] MambaBlock (Mamba1) multi-token Scan support
- [ ] Integration tests for NemotronH
