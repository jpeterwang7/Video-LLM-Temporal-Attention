"""
Bundle the attention files of the labelled videos, so the label analyses can run on a laptop.

Writes <run-dir>/../labelled_attention.tar.gz holding results.jsonl and the .npz attention file of
every video in the given labels files, at the same paths as under <run-dir>. Unpack it inside
outputs/ on the other computer: tar xzf labelled_attention.tar.gz
"""
import argparse
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("labels", type=Path, nargs="+")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/qwen3vl_8b"))
    args = parser.parse_args()

    wanted = set()
    for path in args.labels:
        wanted |= {(l["category"], l["video"]) for l in json.load(open(path))["labels"]}

    out = args.run_dir.parent / "labelled_attention.tar.gz"
    count = 0
    with tarfile.open(out, "w:gz") as tar:
        tar.add(args.run_dir / "results.jsonl", arcname=f"{args.run_dir.name}/results.jsonl")
        with open(args.run_dir / "results.jsonl") as f:
            for line in f:
                row = json.loads(line)
                if (row["category"], row["video"]) in wanted and "error" not in row:
                    tar.add(args.run_dir / row["attention_file"], arcname=f"{args.run_dir.name}/{row['attention_file']}")
                    count += 1
    print(f"Wrote {count} attention files and results.jsonl to {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
