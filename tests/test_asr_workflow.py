from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "workflows" / "qwen3_asr_local.json"
API_PATH = ROOT / "workflows" / "api" / "qwen3_asr_local.api.json"


class AsrWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ui = json.loads(UI_PATH.read_text())
        cls.api = json.loads(API_PATH.read_text())

    def test_api_graph_has_expected_local_asr_path(self) -> None:
        self.assertEqual(self.api["1"]["class_type"], "LoadAudio")
        self.assertEqual(self.api["2"]["class_type"], "Qwen3ASRLoader")
        self.assertEqual(self.api["3"]["class_type"], "Qwen3ASRTranscribe")
        self.assertEqual(self.api["4"]["class_type"], "PreviewAny")
        self.assertEqual(self.api["2"]["inputs"]["repo_id"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(self.api["2"]["inputs"]["source"], "HuggingFace")
        self.assertEqual(self.api["3"]["inputs"]["model"], ["2", 0])
        self.assertEqual(self.api["3"]["inputs"]["audio"], ["1", 0])
        self.assertEqual(self.api["3"]["inputs"]["language"], "auto")
        self.assertFalse(self.api["3"]["inputs"]["return_timestamps"])
        self.assertEqual(self.api["4"]["inputs"]["source"], ["3", 0])

    def test_ui_graph_wires_audio_model_and_transcript_preview(self) -> None:
        nodes = {node["id"]: node for node in self.ui["nodes"]}
        types = {node["type"] for node in nodes.values()}
        self.assertTrue({"LoadAudio", "Qwen3ASRLoader", "Qwen3ASRTranscribe", "PreviewAny"}.issubset(types))
        self.assertEqual(self.ui["last_node_id"], len(nodes))
        self.assertEqual(self.ui["last_link_id"], len(self.ui["links"]))

        links = {link[0]: link for link in self.ui["links"]}
        for node in nodes.values():
            for output in node["outputs"]:
                for link_id in output.get("links") or []:
                    self.assertIn(link_id, links)
            for input_socket in node["inputs"]:
                if input_socket.get("link") is not None:
                    self.assertIn(input_socket["link"], links)

        model = next(node for node in nodes.values() if node["type"] == "Qwen3ASRLoader")
        self.assertEqual(model["widgets_values"][0], "Qwen/Qwen3-ASR-0.6B")
        transcribe = next(node for node in nodes.values() if node["type"] == "Qwen3ASRTranscribe")
        self.assertEqual(transcribe["widgets_values"], ["auto", "", False])

    def test_api_links_resolve_to_existing_nodes(self) -> None:
        for node in self.api.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], self.api)


if __name__ == "__main__":
    unittest.main()
