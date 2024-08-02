# 🛡️ [WIP] Tamper-Resistant Safeguards for Open-Weight LLMs 🤖

## [🛠️ Repo currently under construction 🛠️]

We introduce a novel method, Tampering Attack Resistance (TAR), which is the first defense to withstand a significant number of open-weight fine-tuning attacks on LLMs, while preserving model capabilities.


## Table of Contents

- [📰 Latest News 📰](#-latest-news-)
- [🛡️ What are Tamper-Resistant Safeguards 🛡️](#%EF%B8%8F-what-are-tamper-resistant-safeguards-%EF%B8%8F)
- [🌐 Overview 🌐](#-overview-)
- [☕ Quick Start ☕](#-quick-start-)
  - [⚙️ Installation](#%EF%B8%8F-installation)
  - [🛠️ Running Tamper-Resistance Training](#%EF%B8%8F-running-the-evaluation-pipeline)
  - [➕ Running the Red-teaming evaluation](#➕-running-the-red-teaming-evaluation)
- [⚓ Documentation ⚓](#-documentation-)
- [🗺️ Roadmap 🗺️](#-roadmap-)
- [🙏 Acknowledgement and Citation 🙏](#-acknowledgements-and-citation-)

## 🛡️ What are Tamper-Resistant Safeguards? 🛡️

Tamper-Resistant Safeguards are security measures designed for open-weight large language models (LLMs) to protect against malicious modifications of the model's weights. Unlike traditional safeguards that focus on preventing input-based attacks, these advanced safeguards prevent adversaries with access to full model weights from recovering performance on harmful capabilities. We demonstrate in our extensive red-teaming evaluation that Tamper-Resistant Safeguards created via TAR are the first to be robust to a significant number of open-weight fine-tuning attacks.


## Acknowledgements and Citation
```
@misc{tamirisa2024tamperresistantsafeguardsopenweightllms,
      title={Tamper-Resistant Safeguards for Open-Weight LLMs}, 
      author={Rishub Tamirisa and Bhrugu Bharathi and Long Phan and Andy Zhou and Alice Gatti and Tarun Suresh and Maxwell Lin and Justin Wang and Rowan Wang and Ron Arel and Andy Zou and Dawn Song and Bo Li and Dan Hendrycks and Mantas Mazeika},
      year={2024},
      eprint={2408.00761},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.00761}, 
}
```
