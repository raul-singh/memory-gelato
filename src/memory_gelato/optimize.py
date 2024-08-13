import subprocess as sp
import os
from memory_gelato.estimator import *
from memory_gelato.architectures import model_architecture

QUANTIZATION_LEVELS = [None, 8, 4]

class NotEnoughMemoryError(Exception):
    pass


def get_memory_budget(tolerance=0.025, manual_memory=None):
    available = 1 - tolerance

    if manual_memory:
        return available * manual_memory

    else:
        return get_gpu_memory()[0] * 1024 * 1024 * available

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    free_values = [int(x.split()[0]) for _, x in enumerate(free_info)]
    return free_values


def memory_optimal_config(
    model_id,
    seq_len,
    preferred_batch_size=16,
    min_rank=16,
    max_batch_size=128,
    max_rank=1024,
    quantization="auto",
    available_memory=None,
    return_memory_usage=False
):

    model = model_architecture(model_id)
    budget = get_memory_budget(manual_memory=available_memory)

    lowest_quantization = QUANTIZATION_LEVELS[-1]

    if quantization == "auto":
        for q in QUANTIZATION_LEVELS:
            try:
                memory_config = compute_configuration(
                    model_id,
                    model,
                    seq_len,
                    budget,
                    q,
                    preferred_batch_size,
                    min_rank,
                    max_batch_size,
                    max_rank,
                    return_memory_usage
                )
                break
            except NotEnoughMemoryError as e:
                if q == lowest_quantization:
                    raise e
                else:
                    pass

        q_bit = q

    else:
        q_bit = quantization
        memory_config = compute_configuration(
            model_id,
            model,
            seq_len,
            budget,
            q_bit,
            preferred_batch_size,
            min_rank,
            max_batch_size,
            max_rank,
            return_memory_usage
        )


    memory_config["quantization bit"] = q_bit

    return memory_config


def compute_configuration(
    model_id,
    model,
    seq_len,
    budget,
    quant_bit,
    preferred_batch_size,
    min_rank,
    max_batch_size,
    max_rank,
    return_memory_usage
):

    base_memory = model_memory(model, quant_bit) + static_memory(seq_len)
    grad_acc = 1
    min_trainable_memory = n_trainable_parameters_memory(model, min_rank, grad_acc)

    if base_memory + min_trainable_memory > budget:
        raise NotEnoughMemoryError(f"The base memory of the model {model_id} exceeds the available memory. "\
            "Consider using a smaller model.")

    for b in range(preferred_batch_size, max_batch_size+1):
        batch_memory = batch_dependent_memory(model, model_id, b, seq_len)
        footprint = batch_memory + base_memory + min_trainable_memory

        if footprint > budget:
            batch_size = b - 1
            break

    batch_memory = batch_dependent_memory(model, model_id, batch_size, seq_len)

    if batch_size < preferred_batch_size:
        actual_batch_size = preferred_batch_size

        grad_acc_values = filter(
            lambda x: preferred_batch_size%x == 0, range(1, preferred_batch_size+1)
            )

        for g in grad_acc_values:
            actual_batch_size = preferred_batch_size // g
            batch_memory = batch_dependent_memory(model, model_id, actual_batch_size, seq_len)
            footprint = batch_memory + base_memory + min_trainable_memory

            if footprint < budget:
                grad_acc = g
                break

    batch_size = actual_batch_size
    batch_memory = batch_dependent_memory(model, model_id, batch_size, seq_len)

    for r in range(min_rank, max_rank+1):
        trainable_memory = n_trainable_parameters_memory(model, r, grad_acc)
        footprint = batch_memory + base_memory + trainable_memory
        if footprint > budget:
            break

    rank = r - 1

    if rank < min_rank:
        raise NotEnoughMemoryError("The fine-tuning memory of the model exceeds the available memory "\
            "even at the lowest settings. Consider lowering the minimum rank or use a smaller model.")

    memory_config = {
        "batch size": batch_size,
        "rank": rank,
        "gradient accumulation": grad_acc
    }

    if return_memory_usage:
        config["used memory"] = (
            base_memory +
            batch_memory +
            n_trainable_parameters_memory(model, rank, grad_acc)
        )

    return memory_config