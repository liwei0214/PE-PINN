# PE-PINN: Projection-Enhanced Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

A PyTorch implementation of Projection-Enhanced Physics-Informed Neural Networks for solving multi-dimensional neutron transport eigenvalue problems.

## Overview

PE-PINN integrates mean projection operators into the PINN framework to achieve consistent angular discretization across dimensions. The key innovation is extending mean projection from 1D angular intervals to 2D azimuthal intervals and 3D solid angle regions.

### Features

- **1D Mean Projection**: Angular interval averaging using Gauss-Legendre quadrature
- **2D Mean Projection**: Azimuthal angle interval averaging for S_N quadrature
- **3D Mean Projection**: Solid angle region averaging with Simpson quadrature
- **Adaptive Loss Weighting**: Dynamic adjustment of loss component weights
- **Comprehensive Benchmarks**: Validated on Sood, IAEA PWR, and Takeda benchmarks

## Installation

```bash
git clone https://github.com/liwei0214/PE-PINN.git
cd PE-PINN
pip install -r requirements.txt
```

## Quick Start

```python
from pe_pinn import PEPINN1D, PEPINN2D, PEPINN3D

# 1D Sood benchmark
solver_1d = PEPINN1D(n_angles=8)
results_1d = solver_1d.train(epochs=10000)
print(f"k_eff = {results_1d['k_eff']:.5f}")

# 2D IAEA benchmark
solver_2d = PEPINN2D(sn_order=4)
results_2d = solver_2d.train(epochs=15000)

# 3D Takeda benchmark
solver_3d = PEPINN3D(sn_order=2)
results_3d = solver_3d.train(epochs=12000)
```

## Benchmark Results

| Benchmark | Dimension | k_ref | k_PE-PINN | Error (pcm) |
|-----------|-----------|-------|-----------|-------------|
| Sood | 1D | 1.00000 | 1.00000 | 0.0 |
| IAEA PWR | 2D | 1.02959 | 1.03000 | 39.8 |
| Takeda | 3D | 0.97780 | 0.98000 | 225.0 |

## Method

The mean projection constraint enforces that the angular flux at each discrete direction equals the average over its corresponding angular region:

**1D (Angular Interval)**:
```
ψ(x, μ_i) ≈ (1/Δμ) ∫ ψ(x, μ) dμ  over [μ_l, μ_r]
```

**2D (Azimuthal Interval)**:
```
ψ(x, y, φ_d) ≈ (1/Δφ) ∫ ψ(x, y, φ) dφ  over [φ_l, φ_r]
```

**3D (Solid Angle Region)**:
```
ψ(r, Ω_d) ≈ (1/ΔΩ) ∫ ψ(r, Ω) dΩ  over solid angle region
```

The projection consistency loss is computed using Simpson's rule quadrature.

## Project Structure

```
PE-PINN/
├── pe_pinn.py          # Main implementation
├── README.md           # This file
├── requirements.txt    # Dependencies
├── LICENSE             # MIT License
├── figures/            # Output figures
└── results/            # Output results
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- NumPy >= 1.20
- Matplotlib >= 3.4
- Pandas >= 1.3

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2024pepinn,
  title={Projection-Enhanced Physics-Informed Neural Networks for 
         Multi-Dimensional Nuclear Reactor Criticality Calculations},
  author={Li, Wei and Ma, Yan and Zhu, Meng and Ren, Hong-e and Geng, Shiran},
  journal={Nuclear Engineering and Technology},
  year={2024}
}
```

## References

1. Sood, A., Forster, R.A., Parsons, D.K. (2003). Analytical benchmark test set for criticality code verification. *Progress in Nuclear Energy*, 42(1), 55-106.

2. Argonne Code Center (1977). Benchmark Problem Book. ANL-7416, Suppl. 2.

3. Takeda, T., Ikeda, H. (1991). 3-D neutron transport benchmarks. *Journal of Nuclear Science and Technology*, 28(7), 656-669.

4. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Wei Li - liwei@nefu.edu.cn
- Yan Ma - mayan@mdjnu.edu.cn (Corresponding author)
- Meng Zhu - mengzhu@hrbu.edu.cn (Corresponding author)
