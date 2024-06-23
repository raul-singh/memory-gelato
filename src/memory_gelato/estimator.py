from typing import Optional
from transformers import AutoConfig
from .architectures import model_architecture

def simple_peak_memory_estimation(
    model_id: str,
    batch_size: int,
    max_seq_len: int,
    adapter_type: Optional[str] = None,
    adapter_rank: Optional[int] = None,
    adapter_target_layers: str = "all-linear",
    optimizer: Optional[str] = "PagedAdamW8bit",
    bnb_quantization_bit: Optional[int] = None
    ) -> int:

    if (
        bnb_quantization_bit is not None and
        bnb_quantization_bit != 4 and
        bnb_quantization_bit != 8
    ):
        raise RuntimeError(
            f"{bnb_quantization_bit}-bit quantization is not supported."
            )

    model_arch = model_architecture(model_id)
    model_parameters = model_arch.total_parameters()

    if bnb_quantization_bit is not None:
        model_parameter_memory = (
            model_parameters["linear_layers"] * (bnb_quantization_bit/8) +
            model_parameters["others"] * 4 # Norms are upscaled to FP32
        )

    else:
        model_parameter_memory = (
            model_parameters["linear_layers"] * 2 +
            model_parameters["others"] * 2
        )

    if adapter_type is not None:
        n_adapter_parameters = model_arch.adapter_parameters(
            adapter_rank,
            adapter_target_layers,
            adapter_type
            )
        # Assuming adapter in 16-bit
        model_parameter_memory += (n_adapter_parameters * 2)
        n_trainable_parameters = n_adapter_parameters

    else:
        n_trainable_parameters = (
            model_parameters["linear_layers"] +
            model_parameters["others"]
        )

    # TODO implement different optimizers memory footprint
    optimizer_memory = n_trainable_parameters * 6
    gradient_memory = n_trainable_parameters * 4
    cublas_workspace = 8519680

    # These formulae are assuming gradient checkpointing is used
    decoder_block_act = (
        max_seq_len *
        batch_size *
        model_arch.hidden_size *
        model_arch.n_layers *
        2
    )
    other_act = 8 * max_seq_len * batch_size * model_arch.hidden_size
    cross_entropy_act = 8 * max_seq_len * batch_size * model_arch.vocab_size
    logits = 4 * max_seq_len * batch_size * model_arch.vocab_size
    attention_mask = max_seq_len ** 2 * model_arch.n_layers
    data = max_seq_len * batch_size * 32
    casted_lm_head = model_arch.vocab_size * model_arch.hidden_size * 2

    single_block_activations = (
        # Attention block
        2 * max_seq_len * batch_size * model_arch.n_heads * model_arch.head_size +
        2 * max_seq_len * batch_size * model_arch.n_kv_heads * model_arch.head_size +
        4 * model_arch.n_heads * max_seq_len**2 * batch_size +
        model_arch.n_heads * max_seq_len**2 * batch_size +
        2 * model_arch.n_heads + max_seq_len**2 * batch_size +
        2 * max_seq_len * batch_size * model_arch.n_kv_heads * model_arch.head_size +
        4 * max_seq_len * batch_size * model_arch.hidden_size +
        # MLP block
        2 * max_seq_len * batch_size * model_arch.hidden_size +
        4 * max_seq_len * batch_size * model_arch.ff_intermediate +
        14 * max_seq_len * batch_size * model_arch.ff_intermediate +
        max_seq_len * batch_size * model_arch.hidden_size +
        # Norms
        8 * max_seq_len * batch_size * model_arch.hidden_size
    )

    block_intermediate = (
        # Residuals
        6 * max_seq_len * batch_size * model_arch.hidden_size +
        # MLP intermediate
        16 * model_arch.ff_intermediate * max_seq_len * batch_size
    )

    matmul_reconstruction = 4 * model_arch.ff_intermediate * model_arch.hidden_size

    first_peak = 4 * logits + casted_lm_head
    second_peak = logits + 2 * casted_lm_head
    third_peak = (
        block_intermediate +
        single_block_activations +
        matmul_reconstruction +
        logits
    )

    activation_memory = (
        decoder_block_act +
        other_act +
        cross_entropy_act
    )

    total_memory = (
        model_parameter_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory +
        cublas_workspace +
        max(first_peak, second_peak, third_peak) +
        attention_mask +
        data
    )

    return total_memory