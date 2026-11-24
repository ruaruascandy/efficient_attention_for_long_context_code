# Week-3 preflight

Program length and evidence distance in Qwen tokens (Qwen2.5-Coder tokenizer).

| source bytes | items | tokens mean | min | max | evidence tokens | far gap mean | far gap max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 180 | 203.2 | 180 | 220 | 30.8% | 171.7 | 219 |
| 1024 | 180 | 885.8 | 860 | 908 | 6.9% | 703.2 | 907 |
| 4096 | 180 | 3693.1 | 3670 | 3711 | 1.6% | 2948.8 | 3639 |

## Does a sliding window actually cut the evidence off?

A window of W positions per layer has receptive field 1 + L*(W-1) after L layers,
because information is relayed through depth. The table gives the fraction of items
whose farthest evidence token is *unreachable* from the readout.

Layers = 28  (receptive field W=64 -> 1765 tokens)

| source bytes | W=8 | W=16 | W=32 | W=64 | W=128 | W=256 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 22% | 0% | 0% | 0% | 0% | 0% |
| 1024 | 98% | 91% | 9% | 0% | 0% | 0% |
| 4096 | 100% | 100% | 100% | 94% | 9% | 0% |

Layers = 36  (receptive field W=64 -> 2269 tokens)

| source bytes | W=8 | W=16 | W=32 | W=64 | W=128 | W=256 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0% | 0% | 0% | 0% | 0% | 0% |
| 1024 | 98% | 84% | 0% | 0% | 0% | 0% |
| 4096 | 100% | 100% | 99% | 86% | 0% | 0% |

## Mask geometry (sampled items)

`conn` = allowed (query, key) pairs; `visible` = gold evidence reachable within L layers.
| source bytes | layers | mode | conn | conn frac | gold visible |
|---:|---:|---|---:|---:|---:|
| 256 | 28 | full | 19,936 | 100.0% | 100% |
| 256 | 28 | window=8 | 1,559 | 7.9% | 94% |
| 256 | 28 | window=16 | 3,053 | 15.6% | 100% |
| 256 | 28 | window=32 | 5,851 | 29.8% | 100% |
| 256 | 28 | window=64 | 10,677 | 54.3% | 100% |
| 256 | 28 | window=128 | 17,259 | 87.3% | 100% |
| 256 | 28 | local_global=32:first:4 | 6,510 | 33.2% | 100% |
| 256 | 28 | block=4 | 3,273 | 16.7% | 13% |
| 256 | 28 | gold_union=32 | 13,188 | 68.7% | 100% |
| 256 | 36 | full | 19,936 | 100.0% | 100% |
| 256 | 36 | window=8 | 1,559 | 7.9% | 100% |
| 256 | 36 | window=16 | 3,053 | 15.6% | 100% |
| 256 | 36 | window=32 | 5,851 | 29.8% | 100% |
| 256 | 36 | window=64 | 10,677 | 54.3% | 100% |
| 256 | 36 | window=128 | 17,259 | 87.3% | 100% |
| 256 | 36 | local_global=32:first:4 | 6,510 | 33.2% | 100% |
| 256 | 36 | block=4 | 3,273 | 16.7% | 13% |
| 256 | 36 | gold_union=32 | 13,188 | 68.7% | 100% |
| 1024 | 28 | full | 393,273 | 100.0% | 100% |
| 1024 | 28 | window=8 | 7,061 | 1.8% | 45% |
| 1024 | 28 | window=16 | 14,059 | 3.6% | 73% |
| 1024 | 28 | window=32 | 27,861 | 7.1% | 97% |
| 1024 | 28 | window=64 | 54,699 | 13.9% | 100% |
| 1024 | 28 | window=128 | 105,301 | 26.8% | 100% |
| 1024 | 28 | local_global=32:first:4 | 31,272 | 8.0% | 98% |
| 1024 | 28 | block=4 | 14,639 | 3.7% | 4% |
| 1024 | 28 | gold_union=32 | 76,177 | 19.6% | 100% |
| 1024 | 36 | full | 393,273 | 100.0% | 100% |
| 1024 | 36 | window=8 | 7,061 | 1.8% | 62% |
| 1024 | 36 | window=16 | 14,059 | 3.6% | 84% |
| 1024 | 36 | window=32 | 27,861 | 7.1% | 100% |
| 1024 | 36 | window=64 | 54,699 | 13.9% | 100% |
| 1024 | 36 | window=128 | 105,301 | 26.8% | 100% |
| 1024 | 36 | local_global=32:first:4 | 31,272 | 8.0% | 100% |
| 1024 | 36 | block=4 | 14,639 | 3.7% | 4% |
| 1024 | 36 | gold_union=32 | 76,177 | 19.6% | 100% |
| 4096 | 28 | full | 6,803,302 | 100.0% | 100% |
| 4096 | 28 | window=8 | 29,477 | 0.4% | 13% |
| 4096 | 28 | window=16 | 58,891 | 0.9% | 15% |
| 4096 | 28 | window=32 | 117,525 | 1.7% | 29% |
| 4096 | 28 | window=64 | 234,027 | 3.4% | 55% |
| 4096 | 28 | window=128 | 463,957 | 6.8% | 100% |
| 4096 | 28 | local_global=32:first:4 | 132,144 | 1.9% | 29% |
| 4096 | 28 | block=4 | 74,519 | 1.1% | 0% |
| 4096 | 28 | gold_union=32 | 354,724 | 5.2% | 100% |
| 4096 | 36 | full | 6,803,302 | 100.0% | 100% |
| 4096 | 36 | window=8 | 29,477 | 0.4% | 13% |
| 4096 | 36 | window=16 | 58,891 | 0.9% | 20% |
| 4096 | 36 | window=32 | 117,525 | 1.7% | 36% |
| 4096 | 36 | window=64 | 234,027 | 3.4% | 71% |
| 4096 | 36 | window=128 | 463,957 | 6.8% | 100% |
| 4096 | 36 | local_global=32:first:4 | 132,144 | 1.9% | 36% |
| 4096 | 36 | block=4 | 74,519 | 1.1% | 0% |
| 4096 | 36 | gold_union=32 | 354,724 | 5.2% | 100% |
