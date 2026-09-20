# Third-party notices

## CTR-GCN

This release includes adapted implementations of **Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition**, Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng and Weiming Hu, ICCV 2021. Source: [Uason-Chen/CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), examined at commit `e13d7582e281d06711eeecb380a472b278ae1663`.

Upstream license: **Creative Commons Attribution–NonCommercial 4.0 International**. The unchanged license is retained in `UPSTREAM_LICENSES/CTR-GCN_LICENSE.txt`; its disclaimers and limitations of liability apply. This notice does not imply upstream endorsement.

Paths below are relative to `runtime/project/src/aura_har/`:

| Release file | Upstream reference | Adaptation and changes |
|---|---|---|
| `models/ctrgcn_reference.py` | `model/ctrgcn.py` | Channel-wise relation operator, adaptive graph residual unit, multiscale temporal branches, ten-block schedule and input normalization. Refactored classes/names, typed interfaces, validation, module-list construction, registered graph buffer, explicit feature method and configurable width. |
| `models/ctrgcn.py` | `model/ctrgcn.py` | Earlier simplified CTR-GCN implementation; same channel-wise relation construction and overall block schedule, with simplified temporal layers and a different relation-width rule. It is not interchangeable with the final reference implementation. |
| `data/reference.py` | `feeders/tools.py`, `feeders/bone_pairs.py`, `feeders/feeder_ntu.py` | Crop/resize procedure and NTU stream definitions. Bounds/validation were separated; NumPy Generator, frame-index output, sparse sampling and explicit shape checks were added. |
| `models/graph.py` | `graph/ntu_rgb_d.py`, `graph/tools.py` | NTU bone topology and normalized spatial subsets, reorganized into typed functions with vectorized normalization. |
| `data/ntu.py`, `data/ntu120.py` | Official NTU protocols; also present in upstream preprocessing | Benchmark split constants. The raw sequence parser and body-selection code differ from upstream's SGN-based preprocessing. Listing these protocol constants does not assert copied parser code. |

The source mapping identifies substantive implementation correspondence and conservative adaptation attribution; it is not a claim that every line was copied. Frozen runtime source files remain byte-for-byte unchanged so existing source pins remain verifiable. The modification notice is provided here instead of rewriting historical files.

## Acknowledged upstream lineage

CTR-GCN's README acknowledges [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) as its base and [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch) for preprocessing. These acknowledgements are retained. 2s-AGCN publishes CC BY-NC 4.0; SGN publishes the MIT License, copyright Microsoft Corporation. Copies of the examined license texts accompany this release when present in `UPSTREAM_LICENSES/`.

The release does not vendor the upstream SGN or HCN preprocessing trees. Their research attribution does not establish direct copying of those trees into AURA-KSP. No license is invented for HCN or extended to its separate repository.

## NTU RGB+D / NTU RGB+D 120

The research uses NTU RGB+D 120 provided by ROSE Lab, Nanyang Technological University, Singapore. Cite Shahroudy et al., CVPR 2016, and Liu et al., IEEE TPAMI, DOI [10.1109/TPAMI.2019.2916873](https://doi.org/10.1109/TPAMI.2019.2916873).

Original benchmark data are not included and are not licensed by this project. Obtain access and consult the provider's [access conditions and terms](https://rose1.ntu.edu.sg/dataset/actionRecognition/). Released model parameters, predictions, benchmark identifiers and evaluation summaries document the reported experiment; the project license covers only the contributors' rights in those outputs and does not supersede rights in the benchmark.

## Software dependencies

PyTorch, NumPy, SciPy, pandas, Matplotlib, PyYAML and other installed dependencies retain their own licenses. Dependency installation instructions do not relicense those packages. This artifact does not distribute their complete source trees or binary environments.
