"""Generate random integers with an optional seed from CLI or prompt."""
import argparse
import time
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a list of random integers with an optional seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the RNG. Leave unset to prompt and fall back to current time.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="How many random integers to output (default: 10).",
    )
    return parser.parse_args()


def resolve_seed(seed_arg: int | None) -> int:
    """Return a usable seed from CLI arg, prompt, or current time."""
    default_seed = int(time.time())
    if seed_arg is not None:
        return seed_arg

    try:
        raw = input("Type your seed (empty uses the current time): ").strip()
    except EOFError:
        return default_seed

    if raw:
        return int(raw)
    return default_seed


def main() -> None:
    args = parse_args()
    seed = resolve_seed(args.seed)

    np.random.seed(seed)
    print(f"Seed: {seed}")

    for _ in range(args.count):
        print(np.random.randint(100))


if __name__ == "__main__":
    main()
