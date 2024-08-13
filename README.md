# memory-gelato
:icecream: Memory-GELATO (Memory <ins>G</ins>PU <ins>E</ins>stimation of <ins>L</ins>LM <ins>A</ins>llocation for <ins>T</ins>raining <ins>O</ins>ptimization) is a simple tool for LLM peak training memory estimation and memory constrained hyperparameter search.

## Basic Usage Example

First, the hyperparameters you are going to use that will impact memory must be defined, like batch size, sequence length and model. You can use the `MemoryArgs` dataclass for this.

```python
from memory_gelato import MemoryArgs, simple_peak_estimation

# Here you can set your memory sensitive hyperparameters
memory_args = MemoryArgs(
    model_id="mistralai/Mistral-7B-v0.3",
    batch_size=16,
    max_sequence_length=512,
    adapter_type="lora",
    adapter_rank=16,
    bnb_quantization_bit=4
)
```

After this, to get an estimation of the peak memory footprint, run the method `simple_peak_estimation` passing the `MemoryArgs` defined before.

```python
memory_estimation = simple_peak_estimation(memory_args)
```

## Automatic Memory Optimization Example

GELATO can automatically choose the memory sensitive hyperparameters and quantization level for you automatically. It chooses a configuration that tries to fill the available memory.

To do this, you can use the function `memory_optimal_config`:

```python
from memory_gelato import memory_optimal_config

hyp_configuration = memory_optimal_config(
    model_id="meta-llama/Meta-Llama-3-8B",
    seq_len=1024
)
```

When running this code on a machine equipped with 24 GB of VRAM, the content of `hyp_configuration` is:

```python
{'batch size': 2,
 'rank': 56,
 'gradient accumulation': 8,
 'quantization bit': 8}
```