#!/usr/bin/env python3
"""Build a SeeRepo graph index for SWE-bench instances.

For each instance, this script:
1. Pulls the SWE-bench Docker image for the instance.
2. Copies /testbed out of the container.
3. Builds the repository graph using static AST analysis.
4. Saves the graph as {output_dir}/{instance_id}.pkl.

Usage:
    python scripts/build_graph_index.py \\
        --dataset princeton-nlp/SWE-Bench_Verified \\
        --split test \\
        --output-dir /path/to/graph_index \\
        --workers 8

Requirements:
    pip install datasets networkx
    Docker must be installed and running.
"""

import argparse
import concurrent.futures
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SEEREPO_SRC = Path(__file__).parent.parent / "src"
if str(SEEREPO_SRC) not in sys.path:
    sys.path.insert(0, str(SEEREPO_SRC))


def get_docker_image_name(instance: dict) -> str:
    image_name = instance.get("image_name")
    if image_name is None:
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def build_graph_for_instance(
    instance: dict,
    output_dir: Path,
    fuzzy_search: bool = True,
    global_import: bool = False,
    skip_existing: bool = True,
) -> bool:
    from minisweagent.run.extra.utils.build_graph import build_graph

    instance_id = instance["instance_id"]
    output_pkl = output_dir / f"{instance_id}.pkl"

    if skip_existing and output_pkl.exists():
        logger.info(f"[{instance_id}] already exists, skipping")
        return True

    image_name = get_docker_image_name(instance)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_testbed = Path(tmpdir) / "testbed"
        container_name = f"seerepo_build_{instance_id.replace('__', '_').replace('-', '_')}"

        try:
            logger.info(f"[{instance_id}] pulling image {image_name}")
            subprocess.run(
                ["docker", "pull", image_name],
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                ["docker", "create", "--name", container_name, image_name],
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(f"[{instance_id}] copying /testbed from container")
            subprocess.run(
                ["docker", "cp", f"{container_name}:/testbed", str(local_testbed)],
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"[{instance_id}] docker error: {e.stderr.strip()}")
            return False
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
            )

        if not local_testbed.exists():
            logger.error(f"[{instance_id}] /testbed not found after copy")
            return False

        logger.info(f"[{instance_id}] building graph from {local_testbed}")
        try:
            graph = build_graph(
                str(local_testbed),
                fuzzy_search=fuzzy_search,
                global_import=global_import,
            )
        except Exception as e:
            logger.error(f"[{instance_id}] graph build failed: {e}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_pkl, "wb") as f:
            pickle.dump(graph, f)
        logger.info(f"[{instance_id}] saved to {output_pkl}")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-Bench_Verified", help="HuggingFace dataset path")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--output-dir", required=True, help="Directory to save .pkl files")
    parser.add_argument("--filter", default="", help="Regex filter on instance_id")
    parser.add_argument("--slice", default="", dest="slice_spec", help="Slice spec e.g. '0:50'")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-skip-existing", action="store_true", help="Rebuild even if pkl already exists")
    parser.add_argument("--fuzzy-search", action="store_true", default=True)
    parser.add_argument("--global-import", action="store_true", default=False)
    args = parser.parse_args()

    from datasets import load_dataset
    import re

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset {args.dataset} (split={args.split})")
    instances = list(load_dataset(args.dataset, split=args.split))

    if args.filter:
        instances = [i for i in instances if re.match(args.filter, i["instance_id"])]
    if args.slice_spec:
        parts = [int(x) if x else None for x in args.slice_spec.split(":")]
        instances = instances[slice(*parts)]

    logger.info(f"Building graph index for {len(instances)} instances → {output_dir}")

    skip_existing = not args.no_skip_existing
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                build_graph_for_instance,
                instance,
                output_dir,
                args.fuzzy_search,
                args.global_import,
                skip_existing,
            ): instance["instance_id"]
            for instance in instances
        }
        for future in concurrent.futures.as_completed(futures):
            instance_id = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"[{instance_id}] unexpected error: {e}")

    logger.info(f"Done. {success_count}/{len(instances)} graphs built successfully.")


if __name__ == "__main__":
    main()
