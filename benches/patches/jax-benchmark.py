"""Benchmark script for TTT JAX implementation (forward and backward pass)."""

import argparse
import json
import time

import jax
import jax.numpy as jnp

from ttt.models.model import ModelConfig, CausalLM, CONFIGS
from ttt.models.ttt_layer import TTTLinear, TTTMLP


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTT JAX")
    parser.add_argument("--model-size", type=str, default="125m",
                        choices=["125m", "350m", "760m", "1b"],
                        help="Model size")
    parser.add_argument("--ttt-type", type=str, default="linear",
                        choices=["linear", "mlp"],
                        help="TTT layer type")
    parser.add_argument("--backward", action="store_true",
                        help="Include backward pass (gradient computation)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length")
    parser.add_argument("--mini-batch", type=int, default=16,
                        help="TTT mini-batch size (seq-len must be divisible by this)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of benchmark runs")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type (float32 recommended for ROCm)")
    parser.add_argument("--inner-only", action="store_true",
                        help="Benchmark just the inner TTT layer (single layer)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for scripting)")
    args = parser.parse_args()

    dtype = getattr(jnp, args.dtype)

    # Map model size to config name
    config_name = f"{args.model_size}-TTT"
    ttt_type = f"ttt_{args.ttt_type}"

    # Validate seq_len is divisible by mini_batch
    if args.seq_len % args.mini_batch != 0:
        print(f"ERROR: seq-len ({args.seq_len}) must be divisible by mini-batch ({args.mini_batch})")
        return

    if not args.json:
        print(f"\n{'='*60}")
        print(f"TTT JAX Benchmark ({'Forward + Backward' if args.backward else 'Forward Pass'})")
        print(f"{'='*60}")
        print(f"Model size: {args.model_size}")
        print(f"TTT type: {args.ttt_type}")
        print(f"Batch: {args.batch}, Seq len: {args.seq_len}, Mini-batch: {args.mini_batch}")
        print(f"Device: {jax.devices()[0]}")
        print(f"Dtype: {args.dtype}")
        print(f"{'='*60}\n")

    # Create config
    config_dict = CONFIGS[config_name].copy()
    config_dict["seq_modeling_block"] = ttt_type
    config_dict["mini_batch_size"] = args.mini_batch
    config_dict["max_sequence_length"] = args.seq_len
    config = ModelConfig(**config_dict)

    if not args.json:
        print(f"Hidden size: {config.hidden_size}")
        print(f"Layers: {config.num_hidden_layers}")
        if args.inner_only:
            print("Mode: inner TTT layer only")

    if args.inner_only:
        # Benchmark just the inner TTT layer
        layer_cls = TTTLinear if args.ttt_type == "linear" else TTTMLP
        layer = layer_cls(config=config, dtype=dtype, param_dtype=dtype)

        rng = jax.random.PRNGKey(0)
        hidden_states = jnp.ones((args.batch, args.seq_len, config.hidden_size), dtype=dtype)
        position_ids = jnp.arange(args.seq_len, dtype=jnp.int32)[None, :]

        if not args.json:
            print("\nInitializing layer...")
        variables = layer.init(rng, hidden_states, position_ids=position_ids)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
        if not args.json:
            print(f"Layer parameters: {num_params:,}")

        if args.backward:
            def loss_fn(params, hidden_states, position_ids):
                variables = {"params": params}
                output, _ = layer.apply(variables, hidden_states, position_ids=position_ids)
                return jnp.sum(output)

            @jax.jit
            def train_step(params, hidden_states, position_ids):
                loss, grads = jax.value_and_grad(loss_fn)(params, hidden_states, position_ids)
                return loss, grads

            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                _, grads = train_step(variables["params"], hidden_states, position_ids)
                jax.block_until_ready(grads)

            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            start = time.time()
            for _ in range(args.repeats):
                _, grads = train_step(variables["params"], hidden_states, position_ids)
                jax.block_until_ready(grads)
            avg_time = (time.time() - start) / args.repeats
        else:
            @jax.jit
            def forward(variables, hidden_states, position_ids):
                return layer.apply(variables, hidden_states, position_ids=position_ids)

            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                out, _ = forward(variables, hidden_states, position_ids)
                out.block_until_ready()

            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            start = time.time()
            for _ in range(args.repeats):
                out, _ = forward(variables, hidden_states, position_ids)
                out.block_until_ready()
            avg_time = (time.time() - start) / args.repeats

    else:
        # Full model benchmark (default)

        # Create model
        model = CausalLM(config=config, dtype=dtype, param_dtype=dtype)

        # Initialize
        rng = jax.random.PRNGKey(0)
        input_ids = jnp.ones((args.batch, args.seq_len), dtype=jnp.int32)
        # Target for loss computation (shifted input_ids)
        targets = jnp.roll(input_ids, -1, axis=1)

        if not args.json:
            print("\nInitializing model...")
        variables = model.init(rng, input_ids)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
        if not args.json:
            print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

        if args.backward:
            # Forward + backward pass with gradient computation
            def loss_fn(params, input_ids, targets):
                variables = {"params": params}
                out = model.apply(variables, input_ids)
                # Cross-entropy loss
                logits = out.logits
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                # Gather log probs for targets
                one_hot = jax.nn.one_hot(targets, logits.shape[-1])
                loss = -jnp.sum(log_probs * one_hot) / (args.batch * args.seq_len)
                return loss

            @jax.jit
            def train_step(params, input_ids, targets):
                loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, targets)
                return loss, grads

            # Warmup
            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                loss, grads = train_step(variables["params"], input_ids, targets)
                jax.block_until_ready(grads)

            # Benchmark
            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            start = time.time()
            for _ in range(args.repeats):
                loss, grads = train_step(variables["params"], input_ids, targets)
                jax.block_until_ready(grads)
            avg_time = (time.time() - start) / args.repeats

        else:
            # Forward pass only
            @jax.jit
            def forward(variables, input_ids):
                return model.apply(variables, input_ids)

            # Warmup
            if not args.json:
                print(f"\nWarmup ({args.warmup} runs)...")
            for _ in range(args.warmup):
                out = forward(variables, input_ids)
                out.logits.block_until_ready()

            # Benchmark
            if not args.json:
                print(f"Benchmarking ({args.repeats} runs)...")
            start = time.time()
            for _ in range(args.repeats):
                out = forward(variables, input_ids)
                out.logits.block_until_ready()
            avg_time = (time.time() - start) / args.repeats

    tokens_per_batch = args.batch * args.seq_len
    throughput = tokens_per_batch / avg_time

    if args.json:
        result = {
            "implementation": "jax",
            "model_size": args.model_size,
            "ttt_type": args.ttt_type,
            "backward": args.backward,
            "inner_only": args.inner_only,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "mini_batch": args.mini_batch,
            "dtype": args.dtype,
            "num_params": num_params,
            "time_ms": avg_time * 1000,
            "throughput": throughput,
            "device": str(jax.devices()[0]),
        }
        print(json.dumps(result))
    else:
        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Batch size: {args.batch}")
        print(f"Sequence length: {args.seq_len}")
        print(f"{'Forward + Backward' if args.backward else 'Forward'} time: {avg_time * 1000:.1f}ms")
        print(f"Throughput: {throughput:.1f} tok/s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
