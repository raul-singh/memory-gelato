from typing import Optional
from dataclasses import dataclass
from memory_gelato.architectures import model_architecture, QUADARTIC_ATTENTION_MODELS
from memory_gelato.utils import memory_unit

@dataclass
class MemoryArgs:
    """
    Configuration arguments for memory-efficient fine-tuning of language models.

    Attributes:
        model_id (str):
            The Hugging Face model id.
        batch_size (int):
            The batch size for training.
        gradient_accumulation (int):
            The gradient accumulation value for training.
        max_sequence_length (int):
            The maximum sequence length for input texts.
        optimizer (Optional[str]):
            The optimizer to use, default is "PagedAdamW8bit".
        adapter_type (Optional[str]):
            The type of adapter to use, if any.
        adapter_rank (Optional[int]):
            The rank of the adapter, if applicable.
        adapter_target_layers (str):
            The layers to apply the adapter to, default is "all-linear".
        bnb_quantization_bit (Optional[int]):
            The number of bits for quantization using bitsandbytes, if any.
        double_quantization (bool):
            Whether to use double quantization, default is True.
        quant_block_size (int):
            The block size for quantization, default is 64.
        double_quant_block_size (int):
            The block size for double quantization, default is 256.
    """
    model_id: str

    batch_size: int
    max_sequence_length: int
    gradient_accumulation: int = 1

    optimizer: Optional[str] = "PagedAdamW8bit"

    adapter_type: Optional[str] = None
    adapter_rank: Optional[int] = None
    adapter_target_layers: str = "all-linear"

    bnb_quantization_bit: Optional[int] = None
    double_quantization: bool = True

    quant_block_size: int = 64
    double_quant_block_size: int = 256


def compute_memory(
    training_params: MemoryArgs
    ) -> int:

    if (
        training_params.bnb_quantization_bit is not None and
        training_params.bnb_quantization_bit != 4 and
        training_params.bnb_quantization_bit != 8
    ):
        raise RuntimeError(
            f"{training_params.bnb_quantization_bit}-bit quantization is not supported."
            )

    model_arch = model_architecture(training_params.model_id)
    model_parameters = model_arch.total_parameters()

    batch_size = training_params.batch_size
    seq_length = training_params.max_sequence_length
    grad_acc = training_params.gradient_accumulation
    vocab_size = model_arch.vocab_size
    hidden_size = model_arch.hidden_size
    hidden_mlp = model_arch.ff_intermediate
    n_layers = model_arch.n_layers

    if training_params.bnb_quantization_bit is not None:
        model_parameter_memory = (
            model_parameters["linear_layers"] * (training_params.bnb_quantization_bit/8) +
            model_parameters["others"] * 4 # Norms are upscaled to FP32
        )
        if training_params.double_quantization:
            model_parameter_memory += (
                model_parameters["linear_layers"]/training_params.quant_block_size +
                4 * model_parameters["linear_layers"]/(
                    training_params.quant_block_size *
                    training_params.double_quant_block_size
                    )
            )
        else:
            model_parameter_memory += (
                4 * model_parameters["linear_layers"]/training_params.quant_block_size
            )

    else:
        model_parameter_memory = (
            model_parameters["linear_layers"] * 2 +
            model_parameters["others"] * 2
        )

    if training_params.adapter_type is not None:
        n_adapter_parameters = model_arch.adapter_parameters(
            training_params.adapter_rank,
            training_params.adapter_target_layers,
            training_params.adapter_type
            )
        # Assuming adapter in 16-bit
        model_parameter_memory += (n_adapter_parameters * 2)
        n_trainable_parameters = n_adapter_parameters

    else:
        n_trainable_parameters = (
            model_parameters["linear_layers"] +
            model_parameters["others"]
        )

    # Check if model has quadratic attention enabled by default
    is_quadratic_attention = False
    for model_family in QUADARTIC_ATTENTION_MODELS:
        if model_family in training_params.model_id:
            is_quadratic_attention = True
            break

    # TODO implement different optimizers memory footprint
    optimizer_memory = n_trainable_parameters * 6
    gradient_memory = n_trainable_parameters * 4 * grad_acc
    cublas_workspace = 8519680

    # These formulae are assuming gradient checkpointing is used
    decoder_block_act = (
        seq_length * batch_size * hidden_size * n_layers * 4
    )
    other_act = 8 * seq_length * batch_size * hidden_size
    cross_entropy_act = 8 * seq_length * batch_size * vocab_size
    logits = 4 * seq_length * batch_size * vocab_size
    attention_mask = seq_length ** 2
    data = seq_length * batch_size * 32
    casted_lm_head = vocab_size * hidden_size * 2

    recomputed_activations = (
        # Memory-efficiend SDPA
        42 * seq_length * batch_size * hidden_size +
        # MLP
        18 * seq_length * batch_size * hidden_mlp +
        # Residuals
        8 * seq_length * batch_size * hidden_size +
        # Norms
        8 * seq_length * batch_size * hidden_size
    )

    matmul_reconstruction = 4 * hidden_mlp * hidden_size

    first_peak = 2 * logits + casted_lm_head + cross_entropy_act
    second_peak = logits + 2 * casted_lm_head + logits
    third_peak = (
        recomputed_activations +
        matmul_reconstruction +
        logits
    )
    fourth_peak = (
        4 * 4 * seq_length**2 * batch_size * model_arch.n_heads +
        # Residuals
        1 * 4 * seq_length * batch_size * hidden_mlp +
        # Norms
        2 * 4 * seq_length * batch_size * hidden_size
        + logits
        ) if is_quadratic_attention else 0

    activation_memory = (
        decoder_block_act +
        other_act
    )

    baseline_memory = (
        model_parameter_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory +
        cublas_workspace +
        attention_mask +
        data +
        logits
    )

    peak_memory = max(first_peak, second_peak, third_peak, fourth_peak)

    return (
        baseline_memory,
        peak_memory,
        model_parameter_memory,
        optimizer_memory,
        gradient_memory,
        activation_memory
        )


def simple_peak_estimation(
    training_params: MemoryArgs,
    unit: str = None,
    ) -> int:
    memory = compute_memory(training_params)
    memory = memory[0] + memory[1]
    scale = memory_unit(unit)
    return memory/scale


def model_memory(model, quant_bit=None, double_quant=True, quant_block_size=64, double_quant_block_size=256):

    model_parameters = model.total_parameters()

    if quant_bit is not None:
        model_parameter_memory = (
            model_parameters["linear_layers"] * (quant_bit/8) +
            model_parameters["others"] * 4 # Norms are upscaled to FP32
        )
        if double_quant:
            model_parameter_memory += (
                model_parameters["linear_layers"]/quant_block_size +
                4 * model_parameters["linear_layers"]/(
                    quant_block_size *
                    double_quant_block_size
                    )
            )
        else:
            model_parameter_memory += (
                4 * model_parameters["linear_layers"]/quant_block_size
            )

    else:
        model_parameter_memory = (
            model_parameters["linear_layers"] * 2 +
            model_parameters["others"] * 2
        )

    return model_parameter_memory


def adapter_parameters(model, rank, target_layers="all-linear", adapter_type="l1ra"):
    n_adapter_parameters = model.adapter_parameters(
        rank,
        target_layers,
        adapter_type
    )
    # Assuming adapter in 16-bit
    return n_adapter_parameters

def static_memory(seq_length):
    cublas_workspace = 8519680
    attention_mask = seq_length ** 2

    return cublas_workspace + attention_mask


def largest_peak_memory(model, model_id, batch_size, seq_length):

    batch_size = batch_size
    seq_length = seq_length
    vocab_size = model.vocab_size
    hidden_size = model.hidden_size
    hidden_mlp = model.ff_intermediate
    n_layers = model.n_layers

    is_quadratic_attention = False
    for model_family in QUADARTIC_ATTENTION_MODELS:
        if model_family in model_id:
            is_quadratic_attention = True
            break

    logits = 4 * seq_length * batch_size * vocab_size
    casted_lm_head = vocab_size * hidden_size * 2
    cross_entropy_act = 8 * seq_length * batch_size * vocab_size

    recomputed_activations = (
        # Memory-efficiend SDPA
        42 * seq_length * batch_size * hidden_size +
        # MLP
        18 * seq_length * batch_size * hidden_mlp +
        # Residuals
        8 * seq_length * batch_size * hidden_size +
        # Norms
        8 * seq_length * batch_size * hidden_size
    )

    matmul_reconstruction = 4 * hidden_mlp * hidden_size

    first_peak = 2 * logits + casted_lm_head + cross_entropy_act
    second_peak = logits + 2 * casted_lm_head + logits
    third_peak = (
        recomputed_activations +
        matmul_reconstruction +
        logits
    )
    fourth_peak = (
        4 * 4 * seq_length**2 * batch_size * model.n_heads +
        # Residuals
        1 * 4 * seq_length * batch_size * hidden_mlp +
        # Norms
        2 * 4 * seq_length * batch_size * hidden_size
        + logits
        ) if is_quadratic_attention else 0

    return max(first_peak, second_peak, third_peak, fourth_peak)


def batch_dependent_memory(model, model_id, batch_size, seq_length):
    vocab_size = model.vocab_size
    hidden_size = model.hidden_size
    hidden_mlp = model.ff_intermediate
    n_layers = model.n_layers

    # These formulae are assuming gradient checkpointing is used
    decoder_block_act = (
        seq_length * batch_size * hidden_size * n_layers * 4
    )
    other_act = 8 * seq_length * batch_size * hidden_size

    activation_memory = (
        decoder_block_act +
        other_act
    )

    logits = 4 * seq_length * batch_size * vocab_size
    data = seq_length * batch_size * 32

    peak = largest_peak_memory(model, model_id, batch_size, seq_length)

    return activation_memory + data + logits + peak


def n_trainable_parameters_memory(model, rank, grad_acc, target_layers="all-linear", adapter_type="l1ra"):
    n_adapter_parameters = adapter_parameters(model, rank, target_layers, adapter_type)
    n_trainable_parameters = n_adapter_parameters

    adapter_memory = n_adapter_parameters * 2
    optimizer_memory = n_trainable_parameters * 6 # Assuming paged AdamW 8-bit for the moment
    gradient_memory = n_trainable_parameters * 4 * grad_acc

    return adapter_memory + optimizer_memory + gradient_memory