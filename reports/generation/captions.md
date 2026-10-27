# WP5 image captions

Qwen2.5-VL-7B-Instruct generated captions for every frozen LOEO packet image
using the design spec's Appendix E.5 prompt, deterministic decoding and 48 new-token limit.
Model revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`; quantization: bitsandbytes NF4; prompt SHA-256: `9b30af9bfee374861773122263e7475646e43b1f1f7995f8079e9d0fcad6e565`.
The private cache SHA-256 is `6fc7316f74abca06d0ff52d84f57510991faba65c870837bf36b29c1f88a1030`.
The cache stores verbatim model output. Packet use applies the existing privacy-safe display cleaner.

| Event | Captions | Over 20 words | Longest (words) |
|---|---:|---:|---:|
| hurricane_harvey | 504 | 83 | 30 |
| hurricane_maria | 666 | 119 | 32 |
| mexico_earthquake | 279 | 46 | 32 |

The 20-word instruction is a prompt constraint, not a post-generation truncation rule;
overlength outputs are retained verbatim. Captions are machine generated and have no human accuracy rating.
Private post images and caption text remain outside Git.
