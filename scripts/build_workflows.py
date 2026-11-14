#!/usr/bin/env python3
"""Build the UI and API workflow JSON files in comfy/workflows/.

The generated files are checked in. This script exists so the graphs stay
consistent and are easy to review or change.

Run from anywhere:
    python comfy/scripts/build_workflows.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "workflows"
API_DIR = UI_DIR / "api"

# ---------------------------------------------------------------------------
# Model files. Keep these names in sync with download_models.sh and README.md.
# ---------------------------------------------------------------------------
FL2V = "MiniMax-H3-FL2VA-Pruned-Q3_K_M.gguf"
REF2V = "MiniMax-H3-Ref2VA-Pruned-Q3_K_M.gguf"
QWEN = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
FL2V_TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
REF2V_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
SDXL = "sd_xl_base_1.0.safetensors"

T2V_PROMPT = """A five-second cinematic shot at golden hour: a small white paper boat drifts across a rain puddle on a quiet city street. Warm low sun rakes across wet asphalt, reflections ripple, a light breeze moves fallen leaves. The camera starts low and close to the water, then slowly tracks backward and rises to reveal the street and a distant row of buildings.

One continuous take, no cuts. Shallow depth of field, natural film grain, soft anamorphic highlights.

Audio: gentle rain dripping, distant traffic, a single warm piano note, and a quiet low strings bed. No speech, no music stings."""

I2V_PROMPT = """Animate the provided first frame as a five-second continuous shot. Keep the subject, face, clothing colours, composition and lighting exactly consistent with the input image. Add only subtle natural motion: a slow breath, a blink, fabric moving in a light breeze, and fine environmental movement in the background.

Camera: slow, smooth push-in. No cuts, no morphing, no new objects.

Audio: quiet room tone and soft natural ambience that matches the scene. No speech."""

FLF_PROMPT = """Create a five-second continuous shot that starts exactly on the first frame and ends exactly on the last frame. Keep the subject, face, clothing colours, lighting and overall style consistent between both images. Generate only the natural motion and camera movement needed to travel smoothly from the start composition to the end composition.

One continuous take. No cuts, no morphing, no new objects. Camera: smooth, steady movement. Audio: quiet ambience that matches the scene. No speech."""

R2V_PROMPT = """Use <Picture 1> as the identity reference for the main subject. Keep the same face, hair, body proportions and outfit. Place the subject in a new five-second cinematic shot: they turn toward the camera in a softly lit room, hold eye contact, and give a small, natural smile.

One continuous take. The camera slowly circles a few degrees; shallow depth of field; natural skin texture; soft window light.

Audio: quiet room tone, a faint fabric rustle, and one soft piano note. No speech."""

SDXL_PROMPT = """A calm editorial photograph of a sunlit reading corner, a linen armchair beside a tall window, a small stack of books and a ceramic cup on a wooden side table, soft morning light, muted warm palette, fine film grain, shallow depth of field, natural textures, no text, no people."""

SDXL_NEGATIVE = """text, watermark, logo, signature, low quality, blurry, oversaturated, deformed, extra limbs, cartoon, cgi, plastic skin, harsh flash, cluttered composition"""


# ---------------------------------------------------------------------------
# UI node socket definitions.
#
# Each input is (name, type, optional, label). "widget" inputs are widgets
# promoted to sockets, so they come after normal sockets in the UI.
# ---------------------------------------------------------------------------
W = True  # widget-input marker

UI_IO: dict[str, dict] = {
    "UnetLoaderGGUF": {
        "inputs": [],
        "outputs": [("MODEL", "MODEL")],
    },
    "CLIPLoaderGGUF": {
        "inputs": [],
        "outputs": [("CLIP", "CLIP")],
    },
    "VAELoader": {
        "inputs": [],
        "outputs": [("VAE", "VAE")],
    },
    "LoraLoaderModelOnly": {
        "inputs": [("model", "MODEL", False, None)],
        "outputs": [("MODEL", "MODEL")],
    },
    "ResolutionSelector": {
        "inputs": [],
        "outputs": [("width", "INT"), ("height", "INT")],
    },
    "LoadImage": {
        "inputs": [],
        "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    },
    "ImageScaleToTotalPixels": {
        "inputs": [("image", "IMAGE", False, None)],
        "outputs": [("IMAGE", "IMAGE")],
    },
    "GetImageSize": {
        "inputs": [("image", "IMAGE", False, None)],
        "outputs": [("width", "INT"), ("height", "INT"), ("batch_size", "INT")],
    },
    "CheckpointLoaderSimple": {
        "inputs": [],
        "outputs": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
    },
    "CLIPTextEncode": {
        "inputs": [("clip", "CLIP", False, None)],
        "outputs": [("CONDITIONING", "CONDITIONING")],
    },
    "EmptyLatentImage": {
        "inputs": [],
        "outputs": [("LATENT", "LATENT")],
    },
    "KSampler": {
        "inputs": [
            ("model", "MODEL", False, None),
            ("positive", "CONDITIONING", False, None),
            ("negative", "CONDITIONING", False, None),
            ("latent_image", "LATENT", False, None),
        ],
        "outputs": [("LATENT", "LATENT")],
    },
    "VAEEncode": {
        "inputs": [("pixels", "IMAGE", False, None), ("vae", "VAE", False, None)],
        "outputs": [("LATENT", "LATENT")],
    },
    "VAEDecode": {
        "inputs": [("samples", "LATENT", False, None), ("vae", "VAE", False, None)],
        "outputs": [("IMAGE", "IMAGE")],
    },
    "VAEDecodeAudio": {
        "inputs": [("samples", "LATENT", False, None), ("vae", "VAE", False, None)],
        "outputs": [("AUDIO", "AUDIO")],
    },
    "SaveImage": {
        "inputs": [("images", "IMAGE", False, None)],
        "outputs": [],
    },
    "MiniMaxH3ImageToVideo": {
        "inputs": [
            ("clip", "CLIP", False, None),
            ("vae", "VAE", False, None),
            ("first_frame", "IMAGE", True, None),
            ("last_frame", "IMAGE", True, None),
            ("width", "INT", W, None),
            ("height", "INT", W, None),
        ],
        "outputs": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    },
    "MiniMaxH3ReferenceToVideo": {
        "inputs": [
            ("clip", "CLIP", False, None),
            ("vae", "VAE", True, None),
            ("audio_vae", "VAE", True, None),
            ("ref_images.ref_image_0", "IMAGE", True, "ref_image_0"),
            ("ref_images.ref_image_1", "IMAGE", True, "ref_image_1"),
            ("ref_images.ref_image_2", "IMAGE", True, "ref_image_2"),
            ("ref_videos.ref_video_0", "IMAGE", True, "ref_video_0"),
            ("ref_video_audios.ref_video_audio_0", "AUDIO", True, "ref_video_audio_0"),
            ("ref_audios.ref_audio_0", "AUDIO", True, "ref_audio_0"),
            ("width", "INT", W, None),
            ("height", "INT", W, None),
        ],
        "outputs": [("positive", "CONDITIONING"), ("LATENT", "LATENT")],
    },
    "BasicGuider": {
        "inputs": [
            ("model", "MODEL", False, None),
            ("conditioning", "CONDITIONING", False, None),
        ],
        "outputs": [("GUIDER", "GUIDER")],
    },
    "KSamplerSelect": {
        "inputs": [],
        "outputs": [("SAMPLER", "SAMPLER")],
    },
    "BasicScheduler": {
        "inputs": [("model", "MODEL", False, None)],
        "outputs": [("SIGMAS", "SIGMAS")],
    },
    "RandomNoise": {
        "inputs": [],
        "outputs": [("NOISE", "NOISE")],
    },
    "SamplerCustomAdvanced": {
        "inputs": [
            ("noise", "NOISE", False, None),
            ("guider", "GUIDER", False, None),
            ("sampler", "SAMPLER", False, None),
            ("sigmas", "SIGMAS", False, None),
            ("latent_image", "LATENT", False, None),
        ],
        "outputs": [("output", "LATENT"), ("denoised_output", "LATENT")],
    },
    "CreateVideo": {
        "inputs": [
            ("images", "IMAGE", False, None),
            ("audio", "AUDIO", True, None),
        ],
        "outputs": [("VIDEO", "VIDEO")],
    },
    "SaveVideo": {
        "inputs": [("video", "VIDEO", False, None)],
        "outputs": [("video", "VIDEO")],
    },
}

NODE_SIZE = {
    "UnetLoaderGGUF": (420, 82),
    "CLIPLoaderGGUF": (420, 110),
    "VAELoader": (420, 60),
    "LoraLoaderModelOnly": (420, 90),
    "ResolutionSelector": (280, 170),
    "LoadImage": (430, 500),
    "ImageScaleToTotalPixels": (300, 160),
    "GetImageSize": (260, 100),
    "CheckpointLoaderSimple": (420, 110),
    "CLIPTextEncode": (420, 180),
    "EmptyLatentImage": (320, 130),
    "KSampler": (330, 270),
    "VAEEncode": (230, 60),
    "VAEDecode": (230, 60),
    "VAEDecodeAudio": (230, 60),
    "SaveImage": (430, 450),
    "MiniMaxH3ImageToVideo": (560, 460),
    "MiniMaxH3ReferenceToVideo": (600, 560),
    "BasicGuider": (360, 80),
    "KSamplerSelect": (370, 90),
    "BasicScheduler": (370, 150),
    "RandomNoise": (360, 90),
    "SamplerCustomAdvanced": (250, 390),
    "CreateVideo": (270, 140),
    "SaveVideo": (540, 420),
}


class UIGraph:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.by_key: dict[str, dict] = {}

    def add(
        self,
        key: str,
        node_type: str,
        pos: tuple[int, int],
        widgets: list,
        title: str | None = None,
    ) -> dict:
        io = UI_IO[node_type]
        inputs = []
        for name, typ, optional, label in io["inputs"]:
            item = {"name": name, "type": typ, "link": None}
            if label:
                item["label"] = label
            if optional:
                item["shape"] = 7
            if optional is W:
                item["widget"] = {"name": name}
            inputs.append(item)
        outputs = [{"name": n, "type": t, "links": []} for n, t in io["outputs"]]
        node = {
            "id": len(self.nodes) + 1,
            "type": node_type,
            "pos": list(pos),
            "size": list(NODE_SIZE[node_type]),
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "title": title or node_type,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets,
        }
        self.nodes.append(node)
        self.by_key[key] = node
        return node

    def link(self, src: str, src_slot: int, dst: str, dst_slot: int) -> None:
        a, b = self.by_key[src], self.by_key[dst]
        typ = a["outputs"][src_slot]["type"]
        expected = b["inputs"][dst_slot]["type"]
        if typ != expected:
            raise ValueError(f"{src}[{src_slot}] {typ} != {dst}[{dst_slot}] {expected}")
        link_id = len(self.links) + 1
        self.links.append([link_id, a["id"], src_slot, b["id"], dst_slot, typ])
        a["outputs"][src_slot]["links"].append(link_id)
        b["inputs"][dst_slot]["link"] = link_id

    def dump(self, path: Path, name: str) -> None:
        graph = {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}},
            "version": 0.4,
        }
        path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path.relative_to(ROOT)}  ({name})")


def build_t2v() -> None:
    g = UIGraph()
    g.add("unet", "UnetLoaderGGUF", (60, 60), [FL2V])
    g.add("lora", "LoraLoaderModelOnly", (60, 190), [FL2V_TURBO, 1.0])
    g.add("clip", "CLIPLoaderGGUF", (60, 330), [QWEN, "minimax"])
    g.add("vvae", "VAELoader", (60, 490), [VIDEO_VAE])
    g.add("avae", "VAELoader", (60, 590), [AUDIO_VAE])
    g.add("res", "ResolutionSelector", (60, 700), ["16:9 (Widescreen)", 0.2, 32])
    g.add("h3", "MiniMaxH3ImageToVideo", (560, 60), [T2V_PROMPT, 1344, 768, 124])
    g.add("guider", "BasicGuider", (1200, 60), [])
    g.add("sched", "BasicScheduler", (1200, 190), ["simple", 6, 1])
    g.add("sampler", "KSamplerSelect", (1200, 390), ["res_multistep"])
    g.add("noise", "RandomNoise", (1200, 530), [42, "fixed"])
    g.add("sample", "SamplerCustomAdvanced", (1620, 60), [])
    g.add("dec", "VAEDecode", (1940, 60), [])
    g.add("deca", "VAEDecodeAudio", (1940, 170), [])
    g.add("create", "CreateVideo", (2210, 60), [24, 8])
    g.add("save", "SaveVideo", (2540, 60), ["video/minimax_h3_t2v", "auto", "auto"])
    g.link("unet", 0, "lora", 0)
    g.link("lora", 0, "guider", 0)
    g.link("lora", 0, "sched", 0)
    g.link("clip", 0, "h3", 0)
    g.link("vvae", 0, "h3", 1)
    g.link("res", 0, "h3", 4)
    g.link("res", 1, "h3", 5)
    g.link("h3", 0, "guider", 1)
    g.link("h3", 1, "sample", 4)
    g.link("noise", 0, "sample", 0)
    g.link("guider", 0, "sample", 1)
    g.link("sampler", 0, "sample", 2)
    g.link("sched", 0, "sample", 3)
    g.link("sample", 0, "dec", 0)
    g.link("vvae", 0, "dec", 1)
    g.link("sample", 0, "deca", 0)
    g.link("avae", 0, "deca", 1)
    g.link("dec", 0, "create", 0)
    g.link("deca", 0, "create", 1)
    g.link("create", 0, "save", 0)
    g.dump(UI_DIR / "minimax_h3_t2v_32gb.json", "MiniMax H3 text to video + audio")


def build_i2v() -> None:
    g = UIGraph()
    g.add("unet", "UnetLoaderGGUF", (60, 60), [FL2V])
    g.add("lora", "LoraLoaderModelOnly", (60, 190), [FL2V_TURBO, 1.0])
    g.add("clip", "CLIPLoaderGGUF", (60, 330), [QWEN, "minimax"])
    g.add("vvae", "VAELoader", (60, 490), [VIDEO_VAE])
    g.add("avae", "VAELoader", (60, 590), [AUDIO_VAE])
    g.add("load", "LoadImage", (60, 700), ["example.png", "image"])
    g.add("scale", "ImageScaleToTotalPixels", (540, 700), ["lanczos", 0.2, 32])
    g.add("size", "GetImageSize", (880, 700), [])
    g.add("h3", "MiniMaxH3ImageToVideo", (1200, 60), [I2V_PROMPT, 1344, 768, 124])
    g.add("guider", "BasicGuider", (1840, 60), [])
    g.add("sched", "BasicScheduler", (1840, 190), ["simple", 6, 1])
    g.add("sampler", "KSamplerSelect", (1840, 390), ["res_multistep"])
    g.add("noise", "RandomNoise", (1840, 530), [42, "fixed"])
    g.add("sample", "SamplerCustomAdvanced", (2260, 60), [])
    g.add("dec", "VAEDecode", (2580, 60), [])
    g.add("deca", "VAEDecodeAudio", (2580, 170), [])
    g.add("create", "CreateVideo", (2850, 60), [24, 8])
    g.add("save", "SaveVideo", (3180, 60), ["video/minimax_h3_i2v", "auto", "auto"])
    g.link("unet", 0, "lora", 0)
    g.link("lora", 0, "guider", 0)
    g.link("lora", 0, "sched", 0)
    g.link("load", 0, "scale", 0)
    g.link("scale", 0, "size", 0)
    g.link("scale", 0, "h3", 2)
    g.link("size", 0, "h3", 4)
    g.link("size", 1, "h3", 5)
    g.link("clip", 0, "h3", 0)
    g.link("vvae", 0, "h3", 1)
    g.link("h3", 0, "guider", 1)
    g.link("h3", 1, "sample", 4)
    g.link("noise", 0, "sample", 0)
    g.link("guider", 0, "sample", 1)
    g.link("sampler", 0, "sample", 2)
    g.link("sched", 0, "sample", 3)
    g.link("sample", 0, "dec", 0)
    g.link("vvae", 0, "dec", 1)
    g.link("sample", 0, "deca", 0)
    g.link("avae", 0, "deca", 1)
    g.link("dec", 0, "create", 0)
    g.link("deca", 0, "create", 1)
    g.link("create", 0, "save", 0)
    g.dump(UI_DIR / "minimax_h3_i2v_32gb.json", "MiniMax H3 image to video + audio")


def build_flf2v() -> None:
    g = UIGraph()
    g.add("unet", "UnetLoaderGGUF", (60, 60), [FL2V])
    g.add("lora", "LoraLoaderModelOnly", (60, 190), [FL2V_TURBO, 1.0])
    g.add("clip", "CLIPLoaderGGUF", (60, 330), [QWEN, "minimax"])
    g.add("vvae", "VAELoader", (60, 490), [VIDEO_VAE])
    g.add("avae", "VAELoader", (60, 590), [AUDIO_VAE])
    g.add("first", "LoadImage", (60, 700), ["example.png", "image"])
    g.add("first_scale", "ImageScaleToTotalPixels", (540, 700), ["lanczos", 0.2, 32])
    g.add("size", "GetImageSize", (880, 700), [])
    g.add("last", "LoadImage", (60, 1240), ["example.png", "image"])
    g.add("last_scale", "ImageScaleToTotalPixels", (540, 1240), ["lanczos", 0.2, 32])
    g.add("h3", "MiniMaxH3ImageToVideo", (1200, 60), [FLF_PROMPT, 1344, 768, 124])
    g.add("guider", "BasicGuider", (1840, 60), [])
    g.add("sched", "BasicScheduler", (1840, 190), ["simple", 6, 1])
    g.add("sampler", "KSamplerSelect", (1840, 390), ["res_multistep"])
    g.add("noise", "RandomNoise", (1840, 530), [42, "fixed"])
    g.add("sample", "SamplerCustomAdvanced", (2260, 60), [])
    g.add("dec", "VAEDecode", (2580, 60), [])
    g.add("deca", "VAEDecodeAudio", (2580, 170), [])
    g.add("create", "CreateVideo", (2850, 60), [24, 8])
    g.add("save", "SaveVideo", (3180, 60), ["video/minimax_h3_flf2v", "auto", "auto"])
    g.link("unet", 0, "lora", 0)
    g.link("lora", 0, "guider", 0)
    g.link("lora", 0, "sched", 0)
    g.link("first", 0, "first_scale", 0)
    g.link("first_scale", 0, "size", 0)
    g.link("first_scale", 0, "h3", 2)
    g.link("last", 0, "last_scale", 0)
    g.link("last_scale", 0, "h3", 3)
    g.link("size", 0, "h3", 4)
    g.link("size", 1, "h3", 5)
    g.link("clip", 0, "h3", 0)
    g.link("vvae", 0, "h3", 1)
    g.link("h3", 0, "guider", 1)
    g.link("h3", 1, "sample", 4)
    g.link("noise", 0, "sample", 0)
    g.link("guider", 0, "sample", 1)
    g.link("sampler", 0, "sample", 2)
    g.link("sched", 0, "sample", 3)
    g.link("sample", 0, "dec", 0)
    g.link("vvae", 0, "dec", 1)
    g.link("sample", 0, "deca", 0)
    g.link("avae", 0, "deca", 1)
    g.link("dec", 0, "create", 0)
    g.link("deca", 0, "create", 1)
    g.link("create", 0, "save", 0)
    g.dump(UI_DIR / "minimax_h3_flf2v_32gb.json", "MiniMax H3 first + last frame to video")


def build_r2v() -> None:
    g = UIGraph()
    g.add("unet", "UnetLoaderGGUF", (60, 60), [REF2V])
    g.add("lora", "LoraLoaderModelOnly", (60, 190), [REF2V_TURBO, 1.0])
    g.add("clip", "CLIPLoaderGGUF", (60, 330), [QWEN, "minimax"])
    g.add("vvae", "VAELoader", (60, 490), [VIDEO_VAE])
    g.add("avae", "VAELoader", (60, 590), [AUDIO_VAE])
    g.add("res", "ResolutionSelector", (60, 700), ["16:9 (Widescreen)", 0.2, 32])
    g.add("ref", "LoadImage", (60, 910), ["example.png", "image"])
    g.add("h3", "MiniMaxH3ReferenceToVideo", (560, 60), [R2V_PROMPT, 1344, 768, 124, "match"])
    g.add("guider", "BasicGuider", (1240, 60), [])
    g.add("sched", "BasicScheduler", (1240, 190), ["simple", 4, 1])
    g.add("sampler", "KSamplerSelect", (1240, 390), ["res_multistep"])
    g.add("noise", "RandomNoise", (1240, 530), [42, "fixed"])
    g.add("sample", "SamplerCustomAdvanced", (1660, 60), [])
    g.add("dec", "VAEDecode", (1980, 60), [])
    g.add("deca", "VAEDecodeAudio", (1980, 170), [])
    g.add("create", "CreateVideo", (2250, 60), [24, 8])
    g.add("save", "SaveVideo", (2580, 60), ["video/minimax_h3_r2v", "auto", "auto"])
    g.link("unet", 0, "lora", 0)
    g.link("lora", 0, "guider", 0)
    g.link("lora", 0, "sched", 0)
    g.link("clip", 0, "h3", 0)
    g.link("vvae", 0, "h3", 1)
    g.link("avae", 0, "h3", 2)
    g.link("ref", 0, "h3", 3)
    g.link("res", 0, "h3", 9)
    g.link("res", 1, "h3", 10)
    g.link("h3", 0, "guider", 1)
    g.link("h3", 1, "sample", 4)
    g.link("noise", 0, "sample", 0)
    g.link("guider", 0, "sample", 1)
    g.link("sampler", 0, "sample", 2)
    g.link("sched", 0, "sample", 3)
    g.link("sample", 0, "dec", 0)
    g.link("vvae", 0, "dec", 1)
    g.link("sample", 0, "deca", 0)
    g.link("avae", 0, "deca", 1)
    g.link("dec", 0, "create", 0)
    g.link("deca", 0, "create", 1)
    g.link("create", 0, "save", 0)
    g.dump(UI_DIR / "minimax_h3_r2v_32gb.json", "MiniMax H3 reference image to video")


def build_sdxl_t2i() -> None:
    g = UIGraph()
    g.add("ckpt", "CheckpointLoaderSimple", (60, 60), [SDXL])
    g.add("pos", "CLIPTextEncode", (60, 230), [SDXL_PROMPT], "Positive prompt")
    g.add("neg", "CLIPTextEncode", (60, 460), [SDXL_NEGATIVE], "Negative prompt")
    g.add("empty", "EmptyLatentImage", (60, 700), [1024, 1024, 1])
    g.add("ksampler", "KSampler", (560, 60), [42, 24, 6.0, "dpmpp_2m", "karras", 1.0])
    g.add("vae", "VAEDecode", (960, 60), [])
    g.add("save", "SaveImage", (1250, 60), ["images/sdxl_t2i"])
    g.link("ckpt", 1, "pos", 0)
    g.link("ckpt", 1, "neg", 0)
    g.link("ckpt", 0, "ksampler", 0)
    g.link("pos", 0, "ksampler", 1)
    g.link("neg", 0, "ksampler", 2)
    g.link("empty", 0, "ksampler", 3)
    g.link("ksampler", 0, "vae", 0)
    g.link("ckpt", 2, "vae", 1)
    g.link("vae", 0, "save", 0)
    g.dump(UI_DIR / "sdxl_text_to_image.json", "SDXL text to image")


def build_sdxl_i2i() -> None:
    g = UIGraph()
    g.add("ckpt", "CheckpointLoaderSimple", (60, 60), [SDXL])
    g.add("load", "LoadImage", (60, 230), ["example.png", "image"])
    g.add("scale", "ImageScaleToTotalPixels", (540, 230), ["lanczos", 1.0, 32])
    g.add("pos", "CLIPTextEncode", (60, 790), [SDXL_PROMPT], "Positive prompt")
    g.add("neg", "CLIPTextEncode", (60, 1020), [SDXL_NEGATIVE], "Negative prompt")
    g.add("encode", "VAEEncode", (900, 230), [])
    g.add("ksampler", "KSampler", (1200, 60), [42, 24, 6.0, "dpmpp_2m", "karras", 0.55])
    g.add("vae", "VAEDecode", (1600, 60), [])
    g.add("save", "SaveImage", (1890, 60), ["images/sdxl_i2i"])
    g.link("ckpt", 1, "pos", 0)
    g.link("ckpt", 1, "neg", 0)
    g.link("load", 0, "scale", 0)
    g.link("scale", 0, "encode", 0)
    g.link("ckpt", 2, "encode", 1)
    g.link("ckpt", 0, "ksampler", 0)
    g.link("pos", 0, "ksampler", 1)
    g.link("neg", 0, "ksampler", 2)
    g.link("encode", 0, "ksampler", 3)
    g.link("ksampler", 0, "vae", 0)
    g.link("ckpt", 2, "vae", 1)
    g.link("vae", 0, "save", 0)
    g.dump(UI_DIR / "sdxl_image_to_image.json", "SDXL image to image")


# ---------------------------------------------------------------------------
# API-format graphs (what render.py submits).
# ---------------------------------------------------------------------------
def api_node(class_type: str, inputs: dict, title: str | None = None) -> dict:
    node = {"class_type": class_type, "inputs": inputs}
    if title:
        node["_meta"] = {"title": title}
    return node


def dump_api(name: str, graph: dict) -> None:
    path = API_DIR / f"{name}.api.json"
    path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  (API)")


def build_api_t2v() -> None:
    g = {
        "1": api_node("UnetLoaderGGUF", {"unet_name": FL2V}, "Diffusion model (GGUF)"),
        "2": api_node("LoraLoaderModelOnly", {"model": ["1", 0], "lora_name": FL2V_TURBO, "strength_model": 1.0}, "Turbo LoRA"),
        "3": api_node("CLIPLoaderGGUF", {"clip_name": QWEN, "type": "minimax"}, "Text encoder (GGUF)"),
        "4": api_node("VAELoader", {"vae_name": VIDEO_VAE}, "Video VAE"),
        "5": api_node("VAELoader", {"vae_name": AUDIO_VAE}, "Audio VAE"),
        "6": api_node("MiniMaxH3ImageToVideo", {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": T2V_PROMPT,
            "width": 608, "height": 352, "length": 124,
        }, "Text to video conditioning"),
        "7": api_node("BasicGuider", {"model": ["2", 0], "conditioning": ["6", 0]}, "Guider"),
        "8": api_node("KSamplerSelect", {"sampler_name": "res_multistep"}, "Sampler"),
        "9": api_node("BasicScheduler", {"model": ["2", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}, "Scheduler"),
        "10": api_node("RandomNoise", {"noise_seed": 42}, "Seed"),
        "11": api_node("SamplerCustomAdvanced", {
            "noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0],
            "sigmas": ["9", 0], "latent_image": ["6", 1],
        }, "Sample"),
        "12": api_node("VAEDecode", {"samples": ["11", 0], "vae": ["4", 0]}, "Decode video"),
        "13": api_node("VAEDecodeAudio", {"samples": ["11", 0], "vae": ["5", 0]}, "Decode audio"),
        "14": api_node("CreateVideo", {"images": ["12", 0], "audio": ["13", 0], "fps": 24, "bit_depth": 8}, "Mux video"),
        "15": api_node("SaveVideo", {"video": ["14", 0], "filename_prefix": "video/minimax_h3_t2v", "format": "auto", "codec": "auto"}, "Save video"),
    }
    dump_api("minimax_h3_t2v_32gb", g)


def build_api_i2v() -> None:
    g = {
        "1": api_node("UnetLoaderGGUF", {"unet_name": FL2V}, "Diffusion model (GGUF)"),
        "2": api_node("LoraLoaderModelOnly", {"model": ["1", 0], "lora_name": FL2V_TURBO, "strength_model": 1.0}, "Turbo LoRA"),
        "3": api_node("CLIPLoaderGGUF", {"clip_name": QWEN, "type": "minimax"}, "Text encoder (GGUF)"),
        "4": api_node("VAELoader", {"vae_name": VIDEO_VAE}, "Video VAE"),
        "5": api_node("VAELoader", {"vae_name": AUDIO_VAE}, "Audio VAE"),
        "6": api_node("LoadImage", {"image": "example.png"}, "First frame"),
        "7": api_node("ImageScaleToTotalPixels", {"image": ["6", 0], "upscale_method": "lanczos", "megapixels": 0.2, "resolution_steps": 32}, "Fit first frame to 0.2 MP"),
        "8": api_node("GetImageSize", {"image": ["7", 0]}, "Canvas from image"),
        "9": api_node("MiniMaxH3ImageToVideo", {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": I2V_PROMPT,
            "width": ["8", 0], "height": ["8", 1], "length": 124,
            "first_frame": ["7", 0],
        }, "Image to video conditioning"),
        "10": api_node("BasicGuider", {"model": ["2", 0], "conditioning": ["9", 0]}, "Guider"),
        "11": api_node("KSamplerSelect", {"sampler_name": "res_multistep"}, "Sampler"),
        "12": api_node("BasicScheduler", {"model": ["2", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}, "Scheduler"),
        "13": api_node("RandomNoise", {"noise_seed": 42}, "Seed"),
        "14": api_node("SamplerCustomAdvanced", {
            "noise": ["13", 0], "guider": ["10", 0], "sampler": ["11", 0],
            "sigmas": ["12", 0], "latent_image": ["9", 1],
        }, "Sample"),
        "15": api_node("VAEDecode", {"samples": ["14", 0], "vae": ["4", 0]}, "Decode video"),
        "16": api_node("VAEDecodeAudio", {"samples": ["14", 0], "vae": ["5", 0]}, "Decode audio"),
        "17": api_node("CreateVideo", {"images": ["15", 0], "audio": ["16", 0], "fps": 24, "bit_depth": 8}, "Mux video"),
        "18": api_node("SaveVideo", {"video": ["17", 0], "filename_prefix": "video/minimax_h3_i2v", "format": "auto", "codec": "auto"}, "Save video"),
    }
    dump_api("minimax_h3_i2v_32gb", g)


def build_api_flf2v() -> None:
    g = {
        "1": api_node("UnetLoaderGGUF", {"unet_name": FL2V}, "Diffusion model (GGUF)"),
        "2": api_node("LoraLoaderModelOnly", {"model": ["1", 0], "lora_name": FL2V_TURBO, "strength_model": 1.0}, "Turbo LoRA"),
        "3": api_node("CLIPLoaderGGUF", {"clip_name": QWEN, "type": "minimax"}, "Text encoder (GGUF)"),
        "4": api_node("VAELoader", {"vae_name": VIDEO_VAE}, "Video VAE"),
        "5": api_node("VAELoader", {"vae_name": AUDIO_VAE}, "Audio VAE"),
        "6": api_node("LoadImage", {"image": "example.png"}, "First frame"),
        "7": api_node("ImageScaleToTotalPixels", {"image": ["6", 0], "upscale_method": "lanczos", "megapixels": 0.2, "resolution_steps": 32}, "Fit first frame to 0.2 MP"),
        "8": api_node("GetImageSize", {"image": ["7", 0]}, "Canvas from first frame"),
        "9": api_node("LoadImage", {"image": "example.png"}, "Last frame"),
        "10": api_node("ImageScaleToTotalPixels", {"image": ["9", 0], "upscale_method": "lanczos", "megapixels": 0.2, "resolution_steps": 32}, "Fit last frame to 0.2 MP"),
        "11": api_node("MiniMaxH3ImageToVideo", {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": FLF_PROMPT,
            "width": ["8", 0], "height": ["8", 1], "length": 124,
            "first_frame": ["7", 0], "last_frame": ["10", 0],
        }, "First + last frame conditioning"),
        "12": api_node("BasicGuider", {"model": ["2", 0], "conditioning": ["11", 0]}, "Guider"),
        "13": api_node("KSamplerSelect", {"sampler_name": "res_multistep"}, "Sampler"),
        "14": api_node("BasicScheduler", {"model": ["2", 0], "scheduler": "simple", "steps": 6, "denoise": 1.0}, "Scheduler"),
        "15": api_node("RandomNoise", {"noise_seed": 42}, "Seed"),
        "16": api_node("SamplerCustomAdvanced", {
            "noise": ["15", 0], "guider": ["12", 0], "sampler": ["13", 0],
            "sigmas": ["14", 0], "latent_image": ["11", 1],
        }, "Sample"),
        "17": api_node("VAEDecode", {"samples": ["16", 0], "vae": ["4", 0]}, "Decode video"),
        "18": api_node("VAEDecodeAudio", {"samples": ["16", 0], "vae": ["5", 0]}, "Decode audio"),
        "19": api_node("CreateVideo", {"images": ["17", 0], "audio": ["18", 0], "fps": 24, "bit_depth": 8}, "Mux video"),
        "20": api_node("SaveVideo", {"video": ["19", 0], "filename_prefix": "video/minimax_h3_flf2v", "format": "auto", "codec": "auto"}, "Save video"),
    }
    dump_api("minimax_h3_flf2v_32gb", g)


def build_api_r2v() -> None:
    g = {
        "1": api_node("UnetLoaderGGUF", {"unet_name": REF2V}, "Diffusion model (GGUF)"),
        "2": api_node("LoraLoaderModelOnly", {"model": ["1", 0], "lora_name": REF2V_TURBO, "strength_model": 1.0}, "Turbo LoRA"),
        "3": api_node("CLIPLoaderGGUF", {"clip_name": QWEN, "type": "minimax"}, "Text encoder (GGUF)"),
        "4": api_node("VAELoader", {"vae_name": VIDEO_VAE}, "Video VAE"),
        "5": api_node("VAELoader", {"vae_name": AUDIO_VAE}, "Audio VAE"),
        "6": api_node("LoadImage", {"image": "example.png"}, "Reference image 1"),
        "7": api_node("MiniMaxH3ReferenceToVideo", {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
            "prompt": R2V_PROMPT, "width": 608, "height": 352,
            "length": 124, "ref_image_size": "match",
            "ref_images.ref_image_0": ["6", 0],
        }, "Reference to video conditioning"),
        "8": api_node("BasicGuider", {"model": ["2", 0], "conditioning": ["7", 0]}, "Guider"),
        "9": api_node("KSamplerSelect", {"sampler_name": "res_multistep"}, "Sampler"),
        "10": api_node("BasicScheduler", {"model": ["2", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}, "Scheduler"),
        "11": api_node("RandomNoise", {"noise_seed": 42}, "Seed"),
        "12": api_node("SamplerCustomAdvanced", {
            "noise": ["11", 0], "guider": ["8", 0], "sampler": ["9", 0],
            "sigmas": ["10", 0], "latent_image": ["7", 1],
        }, "Sample"),
        "13": api_node("VAEDecode", {"samples": ["12", 0], "vae": ["4", 0]}, "Decode video"),
        "14": api_node("VAEDecodeAudio", {"samples": ["12", 0], "vae": ["5", 0]}, "Decode audio"),
        "15": api_node("CreateVideo", {"images": ["13", 0], "audio": ["14", 0], "fps": 24, "bit_depth": 8}, "Mux video"),
        "16": api_node("SaveVideo", {"video": ["15", 0], "filename_prefix": "video/minimax_h3_r2v", "format": "auto", "codec": "auto"}, "Save video"),
    }
    dump_api("minimax_h3_r2v_32gb", g)


def build_api_sdxl_t2i() -> None:
    g = {
        "1": api_node("CheckpointLoaderSimple", {"ckpt_name": SDXL}, "SDXL checkpoint"),
        "2": api_node("CLIPTextEncode", {"clip": ["1", 1], "text": SDXL_PROMPT}, "Positive prompt"),
        "3": api_node("CLIPTextEncode", {"clip": ["1", 1], "text": SDXL_NEGATIVE}, "Negative prompt"),
        "4": api_node("EmptyLatentImage", {"width": 1024, "height": 1024, "batch_size": 1}, "Canvas"),
        "5": api_node("KSampler", {
            "model": ["1", 0], "seed": 42, "steps": 24, "cfg": 6.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "denoise": 1.0,
        }, "Sample"),
        "6": api_node("VAEDecode", {"samples": ["5", 0], "vae": ["1", 2]}, "Decode"),
        "7": api_node("SaveImage", {"images": ["6", 0], "filename_prefix": "images/sdxl_t2i"}, "Save image"),
    }
    dump_api("sdxl_text_to_image", g)


def build_api_sdxl_i2i() -> None:
    g = {
        "1": api_node("CheckpointLoaderSimple", {"ckpt_name": SDXL}, "SDXL checkpoint"),
        "2": api_node("CLIPTextEncode", {"clip": ["1", 1], "text": SDXL_PROMPT}, "Positive prompt"),
        "3": api_node("CLIPTextEncode", {"clip": ["1", 1], "text": SDXL_NEGATIVE}, "Negative prompt"),
        "4": api_node("LoadImage", {"image": "example.png"}, "Source image"),
        "5": api_node("ImageScaleToTotalPixels", {"image": ["4", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 32}, "Fit to 1 MP"),
        "6": api_node("VAEEncode", {"pixels": ["5", 0], "vae": ["1", 2]}, "Encode"),
        "7": api_node("KSampler", {
            "model": ["1", 0], "seed": 42, "steps": 24, "cfg": 6.0,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["6", 0], "denoise": 0.55,
        }, "Sample"),
        "8": api_node("VAEDecode", {"samples": ["7", 0], "vae": ["1", 2]}, "Decode"),
        "9": api_node("SaveImage", {"images": ["8", 0], "filename_prefix": "images/sdxl_i2i"}, "Save image"),
    }
    dump_api("sdxl_image_to_image", g)


def main() -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)
    build_t2v()
    build_i2v()
    build_flf2v()
    build_r2v()
    build_sdxl_t2i()
    build_sdxl_i2i()
    build_api_t2v()
    build_api_i2v()
    build_api_flf2v()
    build_api_r2v()
    build_api_sdxl_t2i()
    build_api_sdxl_i2i()


if __name__ == "__main__":
    main()
