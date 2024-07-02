# memory-gelato
Memory-GELATO (Memory <ins>G</ins>PU <ins>E</ins>stimation of <ins>L</ins>LM <ins>A</ins>llocation for <ins>T</ins>raining <ins>O</ins>ptimization) is a simple tool for LLM peak training memory estimation and memory constrained hyperparameter search.

## Basic Usage

First, you have to define the hyperparameters you are going to use that will impact memory, like batch size, sequence length and model. You can use the `MemoryArgs` dataclass for this.

```python
from memory_gelato import MemoryArgs

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

After this, if you want to get an estimation of the peak memory footprint, you can just run the method `simple_peak_estimation` passing the `MemoryArgs` you just defined.

```python
memory_estimation = simple_peak_estimation(memory_args)
```