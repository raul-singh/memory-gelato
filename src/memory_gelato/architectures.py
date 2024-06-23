from dataclasses import dataclass
from typing import Dict, Union
from .utils import num_elements
from transformers import AutoConfig, GemmaConfig

@dataclass
class StandardLLMArchitecture:

    def __init__(self, model_config):
        self.hidden_size = model_config.hidden_size
        self.n_heads = model_config.num_attention_heads
        self.n_kv_heads = model_config.num_key_value_heads
        self.head_size = self.hidden_size/self.n_heads
        self.n_layers = model_config.num_hidden_layers
        self.ff_intermediate = model_config.intermediate_size
        self.vocab_size = model_config.vocab_size

        self.q_proj = (self.hidden_size, self.head_size * self.n_heads)
        self.k_proj = (self.head_size * self.n_kv_heads, self.hidden_size)
        self.v_proj = (self.head_size * self.n_kv_heads, self.hidden_size)
        self.o_proj = (self.hidden_size, self.hidden_size)

        self.gate_proj = (self.ff_intermediate, self.hidden_size)
        self.up_proj = (self.ff_intermediate, self.hidden_size)
        self.down_proj = (self.hidden_size, self.ff_intermediate)

        self.norm = self.hidden_size
        self.wte = (self.vocab_size, self.hidden_size)
        self.lm_head = (self.hidden_size, self.vocab_size)

    def total_parameters(self) -> Dict[str, int]:

        attn = (
            num_elements(self.q_proj)
            + num_elements(self.k_proj)
            + num_elements(self.v_proj)
            + num_elements(self.o_proj)
        )

        mlp = (
            num_elements(self.gate_proj)
            + num_elements(self.down_proj)
            + num_elements(self.up_proj)
        )

        other = (
            num_elements(self.wte)
            + num_elements(self.lm_head)
            + self.norm
        )

        return {
            "linear_layers": int(
                self.n_layers * (attn + mlp)
            ),
            "others": int(
                self.n_layers * 2 * self.norm + other
            )
        }

    def adapter_parameters(
        self,
        rank: int,
        target_layers: Union[str, list[str]],
        adapter_type: str
        ) -> int:
        if adapter_type.lower() in ["adalora", "sora", "l1ra"]:
            additional_vector = True
        else:
            additional_vector = False

        linear_layers = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

        adapter_parameters = 0
        for layer in linear_layers:
            if target_layers == "all-linear" or layer in target_layers:
                matrix_size = getattr(self, layer)
                layer_parameters = self.n_layers * (matrix_size[0] * rank + matrix_size[1] * rank)
                if additional_vector:
                    layer_parameters += (rank * self.n_layers)
                adapter_parameters += layer_parameters

        return int(adapter_parameters)

    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_size: int
    ff_intermediate: int
    vocab_size: int
    n_layers: int

    q_proj: tuple[int, int]
    k_proj: tuple[int, int]
    v_proj: tuple[int, int]
    o_proj: tuple[int, int]

    gate_proj: tuple[int, int]
    up_proj: tuple[int, int]
    down_proj: tuple[int, int]

    norm: int
    wte: tuple[int, int]
    lm_head: tuple[int, int]



def model_architecture(model_id: str) -> StandardLLMArchitecture:
    if type(model_id) != str:
        raise RuntimeError(f"Expecting string for model_id but got {type(model_id)}")

    config = AutoConfig.from_pretrained(model_id)

    if type(config) == GemmaConfig:
        # TODO
        return None

    else:
        return StandardLLMArchitecture(config)