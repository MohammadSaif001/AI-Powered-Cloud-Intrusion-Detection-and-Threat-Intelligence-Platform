# Paper Summary 26: Zero-Day Attack Detection using Autoencoders
**Authors:** Mekdad, Y., et al.
**Title:** A Survey on Zero-Day Attacks and the Challenges of Developing AI-based Intrusion Detection
**Venue:** IEEE Open Journal of the Computer Society, Vol. 3, 2022

---

## Architecture
- **Method**: Variational Autoencoder (VAE) trained only on normal traffic
- **Detection Principle**: Reconstruction error thresholding
  - Normal traffic: low reconstruction error (model learned normal distribution)
  - Zero-day attacks: high reconstruction error (out-of-distribution inputs)
- **Threshold Setting**: 3-sigma rule on normal traffic reconstruction error distribution
- **Dataset**: NSL-KDD + custom zero-day traffic generation

## Key Results
| Zero-Day Category | Detection Rate | FPR |
|---|---|---|
| Simulated novel DoS variants | 96.2% | 3.1% |
| Synthetic network worms | 93.8% | 4.2% |
| Fabricated APT lateral movement | 91.4% | 6.3% |
| Overall average | 93.8% | 4.5% |

## Platform Relevance
- **Zero-Day Component**: Directly validates our Deep Denoising Autoencoder (Agrawal et al. 2022)
  component approach. The unsupervised reconstruction-error paradigm is confirmed as practical for
  novel attack detection without labeled examples
- **Threshold Configuration**: 3-sigma rule used in our autoencoder implementation
- **Limitation Awareness**: 6.3% FPR for APT lateral movement indicates autoencoder alone is
  insufficient for low-FPR APT detection; GNN+GAT component is required for topology-based APT detection

## Limitations
- Generated zero-day samples may not represent real-world novel attacks accurately
- FPR can be high in noisy production environments

## Gap Addressed
- Zero-day detection capability (supports FR-04 Autoencoder row)
