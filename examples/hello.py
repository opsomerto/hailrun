"""Fixture script for hailrun dry-run and manual end-to-end tests.

Importing `sibling` (in the same directory) validates that hailrun's default
code-context sync puts the synced directory on PYTHONPATH correctly.
"""

import argparse

try:
    from sibling import greet
except ImportError:
    greet = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", nargs="?", default=None)
    parser.add_argument("--name", default="world")
    args = parser.parse_args()

    if greet is not None:
        print(greet(args.name))
    else:
        print(f"hello, {args.name} (sibling import failed)")

    if args.shard:
        print(f"input shard: {args.shard}")


if __name__ == "__main__":
    main()
