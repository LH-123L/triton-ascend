# Scan/Sort Operations

| api | Brief Description |
| --- | --- |
| [associative_scan](./associative_scan.md) | Applies combine_fn to each element of the input tensor and the carry value along the specified axis, and updates the carry value |
| [cumprod](./cumprod.md) | Returns the cumulative product of all elements of the input tensor along the specified axis |
| [cumsum](./cumsum.md) | Returns the cumulative sum of all elements of the input tensor along the specified axis |
| [histogram](./histogram.md) | Computes a histogram with num_bins bins based on the input tensor, where each bin has a width of 1 and starts at 0 |
| [sort](./sort.md) | Sorts the tensor along the specified dimension |
| [topk](./topk.md) | Returns the top k largest elements along the specified dimension |

```{toctree}
:maxdepth: 3
:hidden:
associative_scan.md
cumprod.md
cumsum.md
histogram.md
sort.md
topk.md
```
