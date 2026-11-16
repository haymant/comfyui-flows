from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
SCRIPTS = ROOT / "scripts"
HELPERS = SCRIPTS / "_comfy_root.sh"


def load_builder():
    spec = importlib.util.spec_from_file_location("comfy_build_workflows", SCRIPTS / "build_workflows.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuilderTests(unittest.TestCase):
    def test_checked_in_graphs_match_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            module = load_builder()
            module.ROOT = tmp
            module.UI_DIR = tmp / "workflows"
            module.API_DIR = tmp / "workflows" / "api"
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()

            checked_in = sorted(path for path in WORKFLOWS.rglob("*.json"))
            self.assertGreater(len(checked_in), 10)
            for source in checked_in:
                relative = source.relative_to(WORKFLOWS)
                generated = module.UI_DIR / relative
                self.assertTrue(generated.is_file(), f"builder did not generate {relative}")
                self.assertEqual(
                    normalize_doc(json.loads(source.read_text())),
                    normalize_doc(json.loads(generated.read_text())),
                    str(relative),
                )


def normalize_doc(doc: dict) -> dict:
    doc = json.loads(json.dumps(doc))
    if "id" in doc and "revision" in doc:
        doc["id"] = "<generated>"
    return doc


class TtsWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ui = json.loads((WORKFLOWS / "qwen3_tts_local.json").read_text())
        cls.api = json.loads((WORKFLOWS / "api" / "qwen3_tts_local.api.json").read_text())

    def test_api_graph_is_custom_voice_to_save_audio(self) -> None:
        self.assertEqual(self.api["1"]["class_type"], "Qwen3Loader")
        self.assertEqual(self.api["2"]["class_type"], "Qwen3CustomVoice")
        self.assertEqual(self.api["3"]["class_type"], "SaveAudio")
        voice = self.api["2"]["inputs"]
        self.assertEqual(voice["model"], ["1", 0])
        self.assertEqual(voice["language"], "English")
        self.assertEqual(voice["speaker"], "Ryan")
        self.assertIn("quick brown fox", voice["text"])
        self.assertEqual(self.api["3"]["inputs"]["audio"], ["2", 0])
        self.assertEqual(self.api["3"]["inputs"]["filename_prefix"], "audio/qwen3_tts")

    def test_ui_graph_links_are_consistent(self) -> None:
        nodes = {node["id"]: node for node in self.ui["nodes"]}
        self.assertEqual(self.ui["last_node_id"], len(nodes))
        self.assertEqual(self.ui["last_link_id"], len(self.ui["links"]))
        types = {node["type"] for node in nodes.values()}
        self.assertEqual(types, {"Qwen3Loader", "Qwen3CustomVoice", "SaveAudio"})

        links = {link[0]: link for link in self.ui["links"]}
        for node in nodes.values():
            for output in node["outputs"]:
                for link_id in output.get("links") or []:
                    self.assertIn(link_id, links)
            for input_socket in node["inputs"]:
                if input_socket.get("link") is not None:
                    self.assertIn(input_socket["link"], links)


class GraphLinkTests(unittest.TestCase):
    def test_every_api_link_points_at_a_real_node(self) -> None:
        for path in sorted((WORKFLOWS / "api").glob("*.api.json")):
            graph = json.loads(path.read_text())
            for node_id, node in graph.items():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        self.assertIn(value[0], graph, f"{path.name}: node {node_id} -> {value[0]}")


class ShellScriptTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        scripts = sorted(SCRIPTS.glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.home = self.tmp / "home"
        self.app_data = self.tmp / "app-data"
        self.instance = self.home / "ComfyUI-Installs" / "ComfyUI-01"
        (self.instance / "ComfyUI").mkdir(parents=True)
        (self.instance / "ComfyUI" / "main.py").write_text("# fake comfyui\n")
        (self.instance / "ComfyUI" / "models").mkdir()
        (self.app_data / "comfy-procs").mkdir(parents=True)
        (self.app_data / "comfy-procs" / "instance.json").write_text(
            json.dumps({"installPath": str(self.instance), "port": 8199})
        )
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.tmp / "config"),
            "XDG_STATE_HOME": str(self.tmp / "state"),
            "XDG_DATA_HOME": str(self.tmp / "data"),
            "COMFY_DESKTOP_DATA": str(self.app_data),
        }

    def helper(self, expression: str, **extra: str) -> str:
        env = dict(self.env)
        env.update(extra)
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; eval "$2"', "bash", str(HELPERS), expression],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_comfy_root_and_server_use_comfy_procs(self) -> None:
        self.assertEqual(self.helper("comfy_root"), str(self.instance / "ComfyUI"))
        self.assertEqual(self.helper("comfy_server"), "http://127.0.0.1:8199")

    def test_comfy_root_scans_desktop_instance_dirs(self) -> None:
        empty = self.tmp / "no-procs"
        self.assertEqual(
            self.helper("comfy_root", COMFY_DESKTOP_DATA=str(empty)),
            str(self.instance / "ComfyUI"),
        )

    def test_models_dir_and_shared_fallback(self) -> None:
        self.assertEqual(self.helper("comfy_models"), str(self.instance / "ComfyUI" / "models"))
        (self.instance / "ComfyUI" / "models").rmdir()
        (self.home / "ComfyUI-Shared").mkdir()
        self.assertEqual(self.helper("comfy_models"), str(self.home / "ComfyUI-Shared"))
        override = self.tmp / "custom-models"
        self.assertEqual(self.helper("comfy_models", COMFY_MODELS=str(override)), str(override))

    def test_device_override_and_startup_args(self) -> None:
        self.assertEqual(self.helper("comfy_device", COMFY_DEVICE="cpu"), "cpu")
        self.assertEqual(self.helper("comfy_startup_args", COMFY_DEVICE="cpu"), "--cpu --cache-none --disable-smart-memory")
        rocm_args = self.helper("comfy_startup_args", COMFY_DEVICE="rocm")
        self.assertIn("--use-pytorch-cross-attention", rocm_args)
        self.assertNotIn("--cpu", rocm_args)
        cuda_args = self.helper("comfy_startup_args", COMFY_DEVICE="cuda")
        self.assertNotIn("--use-pytorch-cross-attention", cuda_args)


if __name__ == "__main__":
    unittest.main()
