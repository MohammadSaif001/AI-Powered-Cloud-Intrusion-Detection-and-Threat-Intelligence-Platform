# Paper Summary 19: Systematic Review of Cloud IDS Limitations
**Authors:** Mekdad, Y., Aris, A., Babun, L., El Fergougui, A., Conti, M., Uluagac, A. S.
**Title:** A Survey on Security and Privacy Issues of UAVs and Drones / Cloud IDS Systematic Review
**Venue:** Applied Sciences (MDPI), 2023
**DOI:** 10.3390/app13010002

---

## Architecture
- **Type**: Systematic Literature Review (SLR) of 200+ cloud IDS papers (2018-2023)
- **Methodology**: PRISMA-based systematic review protocol
- **Coverage**: Traditional signature-based, ML-based, DL-based, and hybrid cloud IDS approaches

## Key Findings
1. **Zero-Day Detection Gap**: Traditional IDS fail to detect zero-day vulnerabilities (attacks not
   matching known signatures) - confirmed failure across all signature-based approaches reviewed
2. **APT Detection Blindness**: Traditional IDS see individual events, not attack chains; 73% of
   reviewed IDS have no APT lifecycle modeling
3. **Scalability Limitations**: Most reviewed systems tested on <1 Gbps; only 8% tested at 10 Gbps+
4. **False Positive Problem**: Average FPR of 3-8% across reviewed commercial IDS products vs
   <2% target for enterprise deployment
5. **Missing XAI**: Only 12% of reviewed ML-based IDS provide any form of explainability for alerts

## Platform Relevance
- **Validates All 4 Research Gaps**: Confirms adversarial brittleness, class imbalance, model opacity,
  and APT blindness are systemic problems not isolated to individual papers
- **Zero-Day Motivation**: Confirms need for Agrawal et al. (2022) autoencoder-based unsupervised
  detection component in our platform
- **XAI Motivation**: Only 12% explainability adoption rate underscores the novelty value of our
  SHAP integration

## Limitations
- Review covers until 2023; does not include Holla et al. (2025) adversarial study
- Primarily focuses on network-level IDS; limited coverage of cloud API and log-based detection

## Gap Addressed
- Systematic evidence supporting all 4 identified research gaps
