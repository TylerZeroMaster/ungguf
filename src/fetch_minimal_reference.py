import argparse
import json
import struct
import sys
from pathlib import Path

import requests

from common import (
    DEFAULT_CONFIG_FILES,
    ModelMeta,
    TensorMeta,
    model_metadata_file,
)

_HF_BASE = "https://huggingface.co"


def exit_for_status(r: requests.Response, context: str) -> None:
    if not r.ok:
        print(  # noqa: T201
            f"HTTP Error fetching {context}: {r.status_code}\n{r.reason}",
            file=sys.stderr,
        )
        sys.exit(1)


def fetch_shard_header(session: requests.Session, url: str) -> dict[str, TensorMeta]:
    r = session.get(url, headers={"Range": "bytes=0-7"})
    exit_for_status(r, url)
    header_size = struct.unpack("<Q", r.content)[0]

    r = session.get(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
    exit_for_status(r, url)
    raw = json.loads(r.content)

    return {
        key: {"shape": meta["shape"], "dtype": meta["dtype"]}
        for key, meta in raw.items()
        if key != "__metadata__"
    }


def fetch_model_shards(
    session: requests.Session, repo_id: str, revision: str, hf_base: str
) -> set[str]:
    index_url = f"{hf_base}/{repo_id}/resolve/{revision}/model.safetensors.index.json"
    r = session.get(index_url)

    shards: set[str]
    success = 200
    not_found = 404
    if r.status_code == success:
        index = r.json()
        shards = set(index["weight_map"].values())
    elif r.status_code == not_found:
        # single-shard model
        shards = {"model.safetensors"}
    else:
        print(f"HTTP Error ({index_url}): {r.status_code}\n{r.reason}")  # noqa: T201
        sys.exit(1)

    return shards


def fetch_model_shapes(repo_id: str, revision: str, hf_base: str) -> dict[str, TensorMeta]:
    with requests.Session() as session:
        shards = fetch_model_shards(session, repo_id, revision, hf_base)

        shapes: dict[str, TensorMeta] = {}
        for shard in sorted(shards):
            shard_url = f"{hf_base}/{repo_id}/resolve/{revision}/{shard}"
            print(f"Fetching header: {shard}")  # noqa: T201
            shapes.update(fetch_shard_header(session, shard_url))

    return shapes


def fetch_minimal_reference(
    repo_id: str, output_dir: str | Path, *, revision="main", hf_base=_HF_BASE
):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    meta_file = model_metadata_file(output_path)
    shapes = fetch_model_shapes(repo_id, revision, hf_base)
    model_meta: ModelMeta = {"shapes": shapes}
    meta_file.write_text(json.dumps(model_meta))


def fetch_reference_files(
    repo_id: str,
    output_dir: str | Path,
    *,
    revision="main",
    hf_base=_HF_BASE,
    extra_files: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    required_files = ["config.json"]
    files = list(DEFAULT_CONFIG_FILES)
    if extra_files:
        files.extend(extra_files)

    output_path.mkdir(exist_ok=True, parents=True)

    base_url = f"{hf_base}/{repo_id}/resolve/{revision}"
    with requests.Session() as session:
        for name in required_files:
            url = f"{base_url}/{name}"
            dst = output_path / name

            if dst.exists():
                continue

            print(f"Fetching: {name}")  # noqa: T201
            r = session.get(url)
            exit_for_status(r, name)
            dst.write_text(r.text)

        for name in files:
            url = f"{base_url}/{name}"
            dst = output_path / name

            if dst.exists():
                continue

            print(f"Fetching: {name}")  # noqa: T201
            r = session.get(url)
            if not r.ok:
                continue
            dst.write_text(r.text)

    print(f"Downloaded reference files to: {output_dir}")  # noqa: T201


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to safetensors using reference model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Path to base model safetensors directory for name mapping",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Download minimal reference model info from this repo",
    )
    parser.add_argument(
        "--hf-base",
        default=_HF_BASE,
        help="Using a different base for URL for HF api",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Use this revision of the repository",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        help="Download these extra files (specify each file as an option)",
    )

    args = parser.parse_args()

    fetch_minimal_reference(
        output_dir=args.reference_model,
        repo_id=args.repo_id,
        hf_base=args.hf_base,
        revision=args.revision,
    )
    fetch_reference_files(
        output_dir=args.reference_model,
        repo_id=args.repo_id,
        revision=args.revision,
        hf_base=args.hf_base,
        extra_files=args.extra_file,
    )


if __name__ == "__main__":
    main()
