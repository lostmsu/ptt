A tool to quickly check tensors in PyTorch checkpoints.

```pwsh
> ptt ls checkpoint.pt
Tensors: 521
name                  dtype shape     mean       std       min       max    minabs
0.tok_emb.weight      fp32  [256,320]  -9.098E-4 1.1061    -4.7645   4.6105  2.89E-5
...
> ptt check finite checkpoint.pt
Tensors: 521
OK: all floating-point tensors are finite. Checked 27644964 element(s).
```

## Installation

Use https://gitcho.co to install the package:

```pwsh
choco install gh.lostmsu.ptt
```