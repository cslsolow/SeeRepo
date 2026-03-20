#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import base64
import copy
import json
import random
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path
import os
import networkx as nx
from networkx.readwrite import json_graph
from minisweagent.run.extra.utils.build_graph import build_graph
import pickle

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent, ExecutionTimeoutError
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


_GRAPH_PNG_ELIDED = "<graph_png_elided>Graph visualization was shown in the previous turn.</graph_png_elided>"

# Vision API limit (e.g. OpenAI/Doubao): 36M pixels. Use 35M for safety margin.
_MAX_IMAGE_PIXELS = 35_000_000


def _resize_png_to_fit(png_bytes: bytes) -> bytes:
    """Resize PNG to fit within _MAX_IMAGE_PIXELS. Returns original bytes if already within limit."""
    try:
        from PIL import Image
        import io
    except ImportError:
        return png_bytes
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        w, h = img.size
        if w * h <= _MAX_IMAGE_PIXELS:
            return png_bytes
        scale = (_MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resample = getattr(Image.Resampling, "LANCZOS", getattr(Image, "LANCZOS", Image.BICUBIC))
        resized = img.resize((new_w, new_h), resample)
        buf = io.BytesIO()
        resized.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return png_bytes


def _has_image_content(content) -> bool:
    """Check if message content contains image (for vision models)."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def _elide_old_image_messages(self) -> None:
        """Replace image content in old user messages with text placeholder.
        Keeps the image only in the last message (current observation) so the model
        sees it once; subsequent turns get a placeholder to reduce token usage.
        """
        for i, msg in enumerate(self.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not _has_image_content(content):
                continue
            # Keep image in the last message (current observation)
            if i == len(self.messages) - 1:
                continue
            msg["content"] = _GRAPH_PNG_ELIDED

    def query(self) -> dict:
        """Query the model. Elide old image messages before sending to reduce context."""
        self._elide_old_image_messages()
        return super().query()

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
        )
        return super().step()

    def execute_action(self, action: dict) -> dict:
        """Override execute_action to handle graph_visualization commands externally."""
        command = action["action"]
        # Run graph_visualization commands on the host rather than inside the container.
        if "python -m minisweagent.run.extra.utils.graph_visualization" in command:
            # graph_visualization.py resolves paths via env vars (priority: env var > CLI arg > default).
            # If the command explicitly specifies --pkl repo_graph.pkl, replace it with the host absolute path.
            # If no --pkl argument is present, MSWEA_REPO_GRAPH_PKL takes effect automatically.

            # Read env vars from the environment config (stored in self.env.config.env, not os.environ).
            external_pkl_path = None
            instance_dir = None
            if hasattr(self.env, "config") and hasattr(self.env.config, "env"):
                external_pkl_path = self.env.config.env.get("MSWEA_REPO_GRAPH_PKL")
                instance_dir = self.env.config.env.get("MSWEA_INSTANCE_DIR")

            if external_pkl_path:
                # Replace --pkl repo_graph.pkl (with or without quotes) with the absolute host path.
                command = re.sub(r'--pkl\s+["\']?repo_graph\.pkl["\']?', f'--pkl {external_pkl_path}', command)

            cwd = instance_dir if instance_dir else os.getcwd()

            exec_env = os.environ.copy()
            if instance_dir:
                exec_env["MSWEA_INSTANCE_DIR"] = instance_dir
            if external_pkl_path:
                exec_env["MSWEA_REPO_GRAPH_PKL"] = external_pkl_path

            timeout = getattr(self.env.config, "timeout", None) if hasattr(self.env, "config") else None
            timeout = timeout or 600
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=exec_env,
                    timeout=timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                output = {"output": result.stdout, "returncode": result.returncode}
            except subprocess.TimeoutExpired as e:
                output = {"output": e.stdout.decode("utf-8", errors="replace") if e.stdout else "", "returncode": -1}
                raise ExecutionTimeoutError(
                    self.render_template(
                        self.config.timeout_template,
                        action=action,
                        output=output.get("output", ""),
                    )
                )
            
            self.has_finished(output)
            return output | {"action": action["action"]}
        else:
            # Regular commands run inside the Docker container via the parent class.
            return super().execute_action(action)

    def _maybe_attach_graph_png(self, output: dict) -> tuple[list[dict], str]:
        """If the action produced graph PNG records, attach them for the LM."""
        attachments: list[dict] = []
        note = ""
        text = (output.get("output") or "").strip()

        def _parse_graph_records(raw: str) -> list[dict]:
            records: list[dict] = []
            decoder = json.JSONDecoder()
            idx = 0
            while True:
                brace = raw.find("{", idx)
                if brace == -1:
                    break
                try:
                    obj, end = decoder.raw_decode(raw[brace:])
                except json.JSONDecodeError:
                    idx = brace + 1
                    continue
                idx = brace + end
                if isinstance(obj, dict) and ("out_png" in obj or "code_png" in obj):
                    records.append(obj)
            return records

        def _parse_png_paths(raw: str) -> list[str]:
            records = _parse_graph_records(raw)
            if records:
                paths: list[str] = []
                for obj in records:
                    chosen = obj.get("code_png") or obj.get("out_png")
                    if chosen:
                        paths.append(str(chosen))
                return paths
            matches = re.finditer(r'(?P<p>[^\\s"]+\\.png)', raw)
            return [m.group("p") for m in matches]

        png_paths = _parse_png_paths(text)
        if not png_paths:
            return attachments, note

        seen: set[str] = set()
        for raw_path in png_paths:
            png_path = Path(raw_path).expanduser()
            if not png_path.is_absolute():
                instance_dir = os.environ.get("MSWEA_INSTANCE_DIR")
                if instance_dir:
                    png_path = (Path(instance_dir) / png_path).resolve()
                else:
                    png_path = png_path.resolve()
            normalized = str(png_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            if not png_path.exists():
                continue
            try:
                png_bytes = png_path.read_bytes()
                png_bytes = _resize_png_to_fit(png_bytes)
                encoded = base64.b64encode(png_bytes).decode("ascii")
                data_url = f"data:image/png;base64,{encoded}"
            except Exception:
                continue
            attachments.append(
                {
                    "type": "image",
                    "path": str(png_path),
                    "mime": "image/png",
                    "url": data_url,
                    "caption": f"Graph visualization ({png_path.name})",
                }
            )

        if attachments:
            note = "\n<graph_png_embedded>data:image/png;base64,...</graph_png_embedded>"
        return attachments, note

    def _render_text_to_png(self, text: str) -> str | None:
        """Render observation text as a PNG image and return a base64 data URL.

        Uses a dark-background monospace style. Returns None if Pillow is unavailable.
        """
        try:
            import io
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None

        FONT_SIZE = 14
        PADDING = 12
        LINE_HEIGHT = FONT_SIZE + 5
        MAX_LINES = 300
        BG_COLOR = (30, 30, 30)
        TEXT_COLOR = (212, 212, 212)
        IMG_MAX_WIDTH = 1600

        lines = text.expandtabs(4).splitlines()

        if len(lines) > MAX_LINES:
            half = MAX_LINES // 2
            elided = len(lines) - MAX_LINES
            lines = lines[:half] + [f"... [{elided} lines elided] ..."] + lines[-half:]

        font = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        ]:
            if Path(font_path).exists():
                try:
                    font = ImageFont.truetype(font_path, FONT_SIZE)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()

        try:
            char_w = font.getbbox("m")[2] - font.getbbox("m")[0]
        except Exception:
            char_w = int(FONT_SIZE * 0.6)

        max_chars = max((len(line) for line in lines), default=40)
        img_w = min(PADDING * 2 + max_chars * char_w, IMG_MAX_WIDTH)
        img_h = max(PADDING * 2 + len(lines) * LINE_HEIGHT, 80)

        img = Image.new("RGB", (img_w, img_h), BG_COLOR)
        draw = ImageDraw.Draw(img)
        y = PADDING
        for line in lines:
            draw.text((PADDING, y), line, font=font, fill=TEXT_COLOR)
            y += LINE_HEIGHT

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = _resize_png_to_fit(buf.getvalue())
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def get_observation(self, response: dict) -> dict:
        """Execute action, attach graph PNG (if any), and return observation."""
        output = self.execute_action(self.parse_action(response))
        attachments, note = self._maybe_attach_graph_png(output)
        observation_text = self.render_template(self.config.action_observation_template, output=output) + note

        is_submission = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (output.get("output") or "")
        text_as_image: bool = getattr(self.config, "text_as_image", False)

        if attachments:
            content = [
                {"type": "image_url", "image_url": {"url": att.get("url") or att.get("path")}}
                for att in attachments
            ]
            self.add_message("user", content)
        elif text_as_image and not is_submission:
            data_url = self._render_text_to_png(observation_text)
            if data_url:
                self.add_message("user", [{"type": "image_url", "image_url": {"url": data_url}}])
            else:
                # Fall back to plain text when Pillow is unavailable.
                self.add_message("user", observation_text)
        else:
            self.add_message("user", observation_text)

        return output | {"attachments": attachments}


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    # Support Jinja2 template rendering for cwd / env values to allow per-instance path switching.
    for key in ("cwd",):
        if key in env_config and isinstance(env_config[key], str):
            env_config[key] = Template(env_config[key], undefined=StrictUndefined).render(**instance)
    if "env" in env_config and isinstance(env_config["env"], dict):
        env_config["env"] = {
            k: Template(v, undefined=StrictUndefined).render(**instance) if isinstance(v, str) else v
            for k, v in env_config["env"].items()
        }
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))

def resolve_repo_path_from_cwd(config: dict, instance: dict) -> str | None:
    cwd = (config.get("environment", {}) or {}).get("cwd")
    if not cwd:
        return None
    return Template(cwd, undefined=StrictUndefined).render(**instance)

def build_and_save_repo_graph_pkl(
    *,
    instance_dir: Path,
    repo_path: str,
    fuzzy_search: bool = True,
    global_import: bool = False,
):
    instance_dir.mkdir(parents=True, exist_ok=True)
    graph_pkl = instance_dir / "repo_graph.pkl"

    # Skip if already built to avoid redundant computation.
    if graph_pkl.exists():
        return

    G = build_graph(
        repo_path,
        fuzzy_search=fuzzy_search,
        global_import=global_import,
    )

    with open(graph_pkl, "wb") as f:
        pickle.dump(G, f)


def _reset_repo_if_dirty(env: Environment | None, repo_path: str | None, instance_id: str, env_class: str) -> None:
    if not env or not repo_path or env_class != "local":
        return
    check = env.execute("git rev-parse --is-inside-work-tree", cwd=repo_path)
    if check["returncode"] != 0:
        return
    status = env.execute("git status --porcelain", cwd=repo_path)
    if status["returncode"] != 0:
        logger.warning(f"[{instance_id}] git status failed; repo reset skipped")
        return
    if not status["output"].strip():
        return
    reset = env.execute("git reset --hard -q HEAD", cwd=repo_path)
    clean = env.execute("git clean -fdx -q", cwd=repo_path)
    if reset["returncode"] != 0 or clean["returncode"] != 0:
        logger.warning(f"[{instance_id}] git reset/clean failed; repo may be dirty")


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    config_local = copy.deepcopy(config)
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config_local.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    agent = None
    extra_info = None
    env = None
    repo_path = None
    env_class = (config_local.get("environment", {}) or {}).get("environment_class", "docker")

    try:
        graph_cfg = config_local.get("graph", {}) or {}
        index_dir = graph_cfg.get("index_dir")
        if index_dir:
            index_dir = os.path.expandvars(index_dir)
            graph_pkl_path = Path(index_dir) / f"{instance_id}.pkl"
            if not graph_pkl_path.exists():
                logger.warning(f"[{instance_id}] graph index pkl not found: {graph_pkl_path}")
        else:
            graph_pkl_path = instance_dir / "repo_graph.pkl"

        env_vars = {
            "MSWEA_INSTANCE_DIR": str(instance_dir),
            "MSWEA_REPO_GRAPH_PKL": str(graph_pkl_path),
        }
        env_cfg = config_local.setdefault("environment", {}) or {}
        env_cfg.setdefault("env", {}).update(env_vars)

        env = get_sb_environment(config_local, instance)
        if graph_cfg.get("enabled", True):
            if index_dir:
                progress_manager.update_instance_status(instance_id, "Using prebuilt graph index")
            else:
                progress_manager.update_instance_status(instance_id, "Building repo graph")
                repo_path = resolve_repo_path_from_cwd(config_local, instance)
                if repo_path:
                    build_and_save_repo_graph_pkl(
                        instance_dir=instance_dir,
                        repo_path=repo_path,
                        fuzzy_search=graph_cfg.get("fuzzy_search", True),
                        global_import=graph_cfg.get("global_import", False),
                    )
                else:
                    logger.warning(f"[{instance_id}] cannot resolve repo_path from environment.cwd")
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **config_local.get("agent", {}),
        )
        exit_status, result = agent.run(task)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        _reset_repo_if_dirty(env, repo_path, instance_id, env_class)
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("./trajectories", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    p = Path(dataset_path)
    if p.suffix in [".jsonl", ".json"]:
        ds = load_dataset("json", data_files={split: str(p)})
        instances = list(ds[split])
    else:
        instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        preds_text = (output_path / "preds.json").read_text().strip()
        if preds_text:
            try:
                existing_instances = list(json.loads(preds_text).keys())
                logger.info(f"Skipping {len(existing_instances)} existing instances")
                instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse preds.json: {e}. Treating as empty.")
        else:
            logger.info("preds.json exists but is empty, skipping no instances")
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
