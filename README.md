# CSDNet: Synergy of Content and Style: Enhanced Remote Sensing Change Detection via Disentanglement and Refinement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/[insert_arxiv_if_available])  
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/dyzy41/CSDNet)

Official PyTorch implementation of **CSDNet**, a novel bitemporal change detection network that leverages content-style disentanglement and contextual refinement to achieve robust and high-precision remote sensing change detection.

## Paper

**Title:** Synergy of Content and Style: Enhanced Remote Sensing Change Detection via Disentanglement and Refinement

**Authors:** Sijun Dong, Changxin Lu, Siming Fu, Xiaoliang Meng*  
*School of Remote Sensing and Information Engineering, Wuhan University, Wuhan, China*  

**Abstract:**

Bitemporal change detection is often hindered by significant style discrepancies in images, stemming from variations in acquisition time and conditions. To mitigate this, we introduce CSDNet (content–style disentanglement network), a novel bitemporal feature interaction network that leverages content–style disentanglement and a channel gating mechanism. In the feature encoding stage, our Content-Style Disentanglement Module (CSDM) disentangles multi-scale features into content and style components using instance normalization. It then employs a dynamic gating mechanism to selectively preserve style information beneficial for change detection while suppressing background noise. A subsequent feature-level swapping strategy enhances information flow and further aligns the style representations between the bitemporal images. In the decoding stage, the Contextual Content Refiner Module (CCRM) uses a joint channel and spatial gating mechanism to attentively filter and refine the style features. These refined features are then recombined with the content features, enabling a fine-grained delineation of change regions. Extensive experiments on five public datasets—LEVIR-CD, SYSU-CD, S2Looking, WHUCD, and MSRSCD—demonstrate that CSDNet significantly surpasses various state-of-the-art methods in F1-score, IoU, and precision.

The source code and pre-trained weights are available at https://github.com/dyzy41/CSDNet.

## Quantitative Results (Test Set Performance)

| Dataset     | OA    | IoU   | F1    | Recall | Precision |
|-------------|-------|-------|-------|--------|-----------|
| **LEVIR-CD**   | 99.16 | **84.47** | **91.58** | 90.00  | 93.23     |
| **SYSU-CD**    | 92.39 | **71.16** | **83.15** | 79.60  | 87.03     |
| **S2Looking**  | 99.24 | **50.72** | **67.31** | 64.34  | 70.55     |
| **WHUCD**      | 99.56 | **90.88** | **95.22** | 95.12  | 95.33     |
| **MSRSCD**     | 93.07 | **62.01** | **76.55** | 76.38  | 76.73     |

CSDNet consistently achieves top-tier or competitive results across diverse scenarios, particularly excelling in datasets with strong style discrepancies (e.g., seasonal changes in S2Looking) and high-resolution details (e.g., WHUCD).

## Requirements

