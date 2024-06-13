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
        4
    )
    other_act = 8 * max_seq_len * batch_size * model_arch.hidden_size
    cross_entropy_act = 8 * max_seq_len * batch_size * model_arch.vocab_size
    logits = 10 * max_seq_len * batch_size * model_arch.vocab_size
    hidden_states = max_seq_len * batch_size * model_arch.hidden_size * model_arch.n_layers * 2

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
        logits +
        cublas_workspace +
        hidden_states
    )

    return total_memory

