"""多瓦片图集逐帧替换回归：用合成临时 bundle，不读写真实游戏文件。

合成包只包含一张贴图和两条网格帧 Sprite，用来覆盖：
读瓦片、跳过透明占位、同一 UV 只写一次、GIF/PNG 帧目录写回、
BC7 压缩、clip 解析失败时的回退、GUI 候选预览与作品集保存。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageChops, ImageStat
from UnityPy.enums import BuildTarget, ClassIDType, TextureFormat
from UnityPy.export.Texture2DConverter import image_to_texture2d
from UnityPy.files.ObjectReader import ObjectReader
from UnityPy.files.SerializedFile import SerializedFile, SerializedType
from UnityPy.helpers.Tpk import get_typetree_node, UnityVersion
from UnityPy.helpers.TypeTreeHelper import write_typetree
from UnityPy.streams import EndianBinaryWriter

from astral_party_auto.modkit import animation, frame_atlas
from astral_party_auto.modkit.animation import _load_bundle
from astral_party_auto.modkit.clip_atlas import inspect_clip_atlas, replace_clip_atlas_checked
from astral_party_auto.mod_controller import ModController
from astral_party_auto.web_app import DesktopApi

UNITY_VERSION = "2022.3.62f3"
TEXTURE_NAME = "Idle-001"
ANIM_NAME = "Idle"
ATLAS_SIZE = (128, 64)
FRAME_SIZE = (64, 48)
FRAME_COUNT = 2


def _dump_typetree(value: dict, node) -> bytes:
    # Unity 序列化数据是小端；EndianBinaryWriter 默认大端，必须显式指定。
    writer = EndianBinaryWriter(endian="<")
    write_typetree(value, node, writer)
    return writer.bytes


def _texture_value(name: str, image: Image.Image, *, bc7: bool) -> dict:
    if bc7:
        raw, texture_format = image_to_texture2d(image, TextureFormat.BC7)
    else:
        raw, texture_format = image.tobytes("raw", "RGBA"), TextureFormat.RGBA32
    return {
        "m_Name": name,
        "m_ForcedFallbackFormat": 4,
        "m_DownscaleFallback": False,
        "m_IsAlphaChannelOptional": False,
        "m_Width": image.width,
        "m_Height": image.height,
        "m_CompleteImageSize": len(raw),
        "m_MipsStripped": 0,
        "m_TextureFormat": int(texture_format),
        "m_MipCount": 1,
        "m_IsReadable": False,
        "m_IsPreProcessed": False,
        "m_IgnoreMipmapLimit": False,
        "m_MipmapLimitGroupName": "",
        "m_StreamingMipmaps": False,
        "m_StreamingMipmapsPriority": 0,
        "m_ImageCount": 1,
        "m_TextureDimension": 2,
        "m_TextureSettings": {"m_FilterMode": 1, "m_Aniso": 1, "m_MipBias": 0.0,
                              "m_WrapU": 1, "m_WrapV": 1, "m_WrapW": 1},
        "m_LightmapFormat": 0,
        "m_ColorSpace": 1,
        "m_PlatformBlob": [],
        "image data": raw,
        "m_StreamData": {"offset": 0, "size": 0, "path": ""},
    }


def _sprite_value(name: str, index: int, texture_path_id: int) -> dict:
    # 每帧占图集的一条 64×48 横带，再由左右两个 32×48 瓦片拼成。
    region = index * FRAME_SIZE[0]
    left = region / ATLAS_SIZE[0]
    middle = (region + FRAME_SIZE[0] / 2) / ATLAS_SIZE[0]
    right = (region + FRAME_SIZE[0]) / ATLAS_SIZE[0]
    bottom = (ATLAS_SIZE[1] - FRAME_SIZE[1]) / ATLAS_SIZE[1]
    quads = [
        ((0.0, 0.0, 32.0, 48.0), (left, bottom, middle, 1.0)),
        ((32.0, 0.0, 64.0, 48.0), (middle, bottom, right, 1.0)),
    ]
    vertices, uvs, index_buffer = [], [], []
    for number, ((x0, y0, x1, y1), (u0, v0, u1, v1)) in enumerate(quads):
        base = number * 4
        vertices += [(x0, y0, 0.0), (x0, y1, 0.0), (x1, y1, 0.0), (x1, y0, 0.0)]
        uvs += [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
        index_buffer += [base, 0, base + 1, 0, base + 2, 0,
                         base + 2, 0, base + 3, 0, base, 0]
    data = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    data += b"".join(struct.pack("<2f", *uv) for uv in uvs)
    return {
        "m_Name": name,
        "m_Rect": {"x": 0.0, "y": 0.0, "width": 64.0, "height": 48.0},
        "m_Offset": {"x": 0.0, "y": 0.0},
        "m_Border": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
        "m_PixelsToUnits": 1.0,
        "m_Pivot": {"x": 0.5, "y": 0.5},
        "m_Extrude": 0,
        "m_IsPolygon": False,
        "m_RenderDataKey": [{"data[0]": index + 1, "data[1]": 2, "data[2]": 3, "data[3]": 4}, index + 5],
        "m_AtlasTags": [],
        "m_SpriteAtlas": {"m_FileID": 0, "m_PathID": 0},
        "m_RD": {
            "texture": {"m_FileID": 0, "m_PathID": texture_path_id},
            "alphaTexture": {"m_FileID": 0, "m_PathID": 0},
            "secondaryTextures": [],
            "m_SubMeshes": [{
                "firstByte": 0, "indexCount": len(index_buffer), "topology": 0,
                "baseVertex": 0, "firstVertex": 0, "vertexCount": 8,
                "localAABB": {"m_Center": {"x": 0.0, "y": 0.0, "z": 0.0},
                              "m_Extent": {"x": 32.0, "y": 24.0, "z": 0.0}},
            }],
            "m_IndexBuffer": index_buffer,
            "m_VertexData": {
                "m_VertexCount": 8,
                "m_Channels": [
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 3},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 1, "offset": 0, "format": 0, "dimension": 2},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                    {"stream": 0, "offset": 0, "format": 0, "dimension": 0},
                ],
                "m_DataSize": data,
            },
            "m_Bindpose": [],
            "textureRect": {"x": 0.0, "y": 0.0, "width": 64.0, "height": 48.0},
            "textureRectOffset": {"x": 0.0, "y": 0.0},
            "atlasRectOffset": {"x": -1.0, "y": -1.0},
            "settingsRaw": 128,
            "uvTransform": {"x": 1.0, "y": 0.0, "z": 1.0, "w": 0.0},
            "downscaleMultiplier": 1.0,
        },
        "m_PhysicsShape": [],
        "m_Bones": [],
    }


def _serialized_file(objects: list[tuple[int, int, dict, object]], version: UnityVersion) -> bytes:
    sf = object.__new__(SerializedFile)
    sf.name = "CAB-synthetic"
    sf.parent = None
    sf.header = SimpleNamespace(version=22, endian="<", reserved=b"\x00\x00\x00")
    sf.unity_version = UNITY_VERSION
    sf.version = version
    sf.target_platform = BuildTarget.StandaloneWindows64
    sf._m_target_platform = int(BuildTarget.StandaloneWindows64)
    # 不写类型树：读取时由 UnityPy 自带的 TPK 定义解析，包更小且更接近无类型树的真实包。
    sf._enable_type_tree = False
    sf.script_types = []
    sf.externals = []
    sf.ref_types = []
    sf.userInformation = ""
    sf.big_id_enabled = 0
    sf.unknown = 0
    sf._cache = {}

    types = []
    for _path_id, class_id, _value, _node in objects:
        typ = SerializedType.__new__(SerializedType)
        typ.__attrs_init__(class_id)
        typ.is_stripped_type = False
        typ.script_type_index = -1
        typ.script_id = b"\x00" * 16
        typ.old_type_hash = b"\x00" * 16
        typ.node = None
        types.append(typ)
    sf.types = types

    readers = {}
    for type_id, (_path_id, class_id, value, node) in enumerate(objects):
        raw = _dump_typetree(value, node)
        readers[_path_id] = ObjectReader(
            assets_file=sf, reader=None, path_id=_path_id, type_id=type_id,
            serialized_type=types[type_id], class_id=class_id,
            type=ClassIDType(class_id), byte_start=0, byte_size=len(raw),
            is_destroyed=None, is_stripped=None, data=raw,
        )
    sf.objects = readers
    return sf.save()


def build_synthetic_bundle(path: str | Path, *, bc7: bool = False,
                           frames: int = FRAME_COUNT) -> Path:
    """生成最小可读的合成包：一张图集 + 若干网格帧 Sprite。"""
    version = UnityVersion.from_str(UNITY_VERSION)
    atlas = Image.new("RGBA", ATLAS_SIZE, (0, 0, 0, 0))
    for index in range(frames):
        region = index * FRAME_SIZE[0]
        for x in range(region, region + FRAME_SIZE[0]):
            for y in range(ATLAS_SIZE[1] - FRAME_SIZE[1], ATLAS_SIZE[1]):
                atlas.putpixel((x, y), (40 + index * 40, 80, 120, 255))
    texture = _texture_value(TEXTURE_NAME, atlas, bc7=bc7)
    objects: list[tuple[int, int, dict, object]] = [
        (1, 28, texture, get_typetree_node(28, version)),
    ]
    for index in range(frames):
        sprite = _sprite_value(f"Idle_{index:05d}", index, 1)
        objects.append((10 + index, 213, sprite, get_typetree_node(213, version)))
    target = Path(path)
    target.write_bytes(_serialized_file(objects, version))
    return target


def _frame_png(path: Path, left: tuple, right: tuple) -> Path:
    image = Image.new("RGBA", FRAME_SIZE, left)
    for x in range(FRAME_SIZE[0] // 2, FRAME_SIZE[0]):
        for y in range(FRAME_SIZE[1]):
            image.putpixel((x, y), right)
    image.save(path)
    return path


def _gif(path: Path, colors: list[tuple]) -> Path:
    frames = [Image.new("RGBA", FRAME_SIZE, color) for color in colors]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


def _png_folder(root: Path, colors: list[tuple]) -> Path:
    folder = root / "frames"
    folder.mkdir()
    for index, color in enumerate(colors):
        Image.new("RGBA", FRAME_SIZE, color).save(folder / f"frame_{index:03d}.png")
    return folder


def _atlas_pixels(path: str | Path) -> Image.Image:
    env = _load_bundle(path)
    data = frame_atlas._texture_of(env, TEXTURE_NAME).read()
    return data.image.convert("RGBA")


class FrameAtlasChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jixing-frame-atlas-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = build_synthetic_bundle(self.root / "synthetic.bundle")
        self.baseline = hashlib.sha256(self.bundle.read_bytes()).hexdigest()
        self.frame = _frame_png(self.root / "frame.png", (255, 0, 0, 255), (0, 0, 255, 255))
        self.gif = _gif(self.root / "frames.gif", [(255, 0, 0, 255), (0, 0, 255, 255)])
        self.folder = _png_folder(self.root, [(0, 255, 0, 255), (255, 255, 0, 255)])

    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.bundle.read_bytes()).hexdigest(), self.baseline,
                         "合成源包被修改")

    def test_inspect_reads_tiles_without_animation_clip(self):
        info = frame_atlas.inspect_texture(self.bundle, TEXTURE_NAME)
        self.assertEqual(info["frames"], FRAME_COUNT)
        self.assertEqual(info["tiles"], FRAME_COUNT * 2)
        self.assertEqual(info["atlas"], ATLAS_SIZE)
        self.assertEqual(info["frame_sizes"], [FRAME_SIZE])

    def test_compose_skips_transparent_placeholder_and_writes_once(self):
        frames = frame_atlas.read_frames(_load_bundle(self.bundle), TEXTURE_NAME)
        original = Image.new("RGBA", ATLAS_SIZE, (20, 20, 20, 255))
        # 把第二帧右半瓦片对应的图集区域设成透明，模拟打包器占位格。
        region = FRAME_SIZE[0] + FRAME_SIZE[0] // 2
        for x in range(region, region + FRAME_SIZE[0] // 2):
            for y in range(FRAME_SIZE[1]):
                original.putpixel((x, y), (0, 0, 0, 0))
        atlas = frame_atlas.compose(frames, [Image.new("RGBA", FRAME_SIZE, (255, 0, 0, 255))],
                                    ATLAS_SIZE, original)
        self.assertEqual(atlas.getpixel((region + 4, 40))[3], 0, "透明占位格被写入")

        shared = frame_atlas.Frame("a", 0, (0.0, 0.0, 64.0, 48.0),
                                   (frame_atlas.Quad((0.0, 0.0, 64.0, 48.0), (0.0, 0.0, 0.5, 0.5)),))
        later = frame_atlas.Frame("b", 1, (0.0, 0.0, 64.0, 48.0),
                                  (frame_atlas.Quad((0.0, 0.0, 64.0, 48.0), (0.0, 0.0, 0.5, 0.5)),))
        composed = frame_atlas.compose([shared, later],
                                       [Image.new("RGBA", FRAME_SIZE, (255, 0, 0, 255)),
                                        Image.new("RGBA", FRAME_SIZE, (0, 0, 255, 255))], ATLAS_SIZE)
        self.assertEqual(composed.getpixel((8, 40)), (255, 0, 0, 255), "同一 UV 被后一帧覆盖")

    def test_replacement_preserves_frames_size_format_and_pixels(self):
        out = self.root / "out.bundle"
        result = frame_atlas.replace_texture_frames(self.bundle, TEXTURE_NAME, [self.frame], out)
        self.assertEqual(result["frames"], FRAME_COUNT)
        self.assertTrue(out.is_file())
        info = frame_atlas.inspect_texture(out, TEXTURE_NAME)
        self.assertEqual(info["frames"], FRAME_COUNT)
        self.assertEqual(info["atlas"], ATLAS_SIZE)
        atlas = _atlas_pixels(out)
        self.assertEqual(atlas.getpixel((8, 40)), (255, 0, 0, 255))
        self.assertEqual(atlas.getpixel((40, 40)), (0, 0, 255, 255))
        self.assertEqual(atlas.getpixel((72, 40)), (255, 0, 0, 255))
        self.assertEqual(atlas.getpixel((104, 40)), (0, 0, 255, 255))

    def test_gif_and_png_directory_use_all_target_frames(self):
        gif_out = self.root / "gif.bundle"
        folder_out = self.root / "folder.bundle"
        gif_result = frame_atlas.replace_animation_frames(self.bundle, ANIM_NAME, self.gif, gif_out)
        folder_result = frame_atlas.replace_animation_frames(self.bundle, ANIM_NAME, self.folder, folder_out)
        self.assertEqual(gif_result["frames"], FRAME_COUNT)
        self.assertEqual(gif_result["source_frames"], 2)
        self.assertEqual(folder_result["frames"], FRAME_COUNT)
        preview = frame_atlas.read_frames_preview(gif_out, ANIM_NAME)
        self.assertEqual(preview["total"], FRAME_COUNT)
        self.assertEqual(len(preview["frames"]), FRAME_COUNT)
        first = Image.open(io.BytesIO(base64.b64decode(preview["frames"][0].split(",", 1)[1]))).convert("RGBA")
        self.assertEqual(first.getpixel((32, 24)), (255, 0, 0, 255))
        adapted = frame_atlas.read_adapted_frames(self.bundle, ANIM_NAME, self.gif)
        self.assertTrue(adapted["frame_atlas"])
        self.assertEqual(len(adapted["frames"]), FRAME_COUNT)

    def test_whole_atlas_import_still_works(self):
        full = self.root / "full-atlas.png"
        Image.new("RGBA", ATLAS_SIZE, (1, 2, 3, 255)).save(full)
        out = self.root / "full.bundle"
        name = replace_clip_atlas_checked(self.bundle, ANIM_NAME, full, out)
        self.assertEqual(name, TEXTURE_NAME)
        self.assertTrue(out.is_file())
        self.assertEqual(frame_atlas.inspect_texture(out, TEXTURE_NAME)["frames"], FRAME_COUNT)

    def test_clip_atlas_and_animation_fall_back_to_frames(self):
        details = inspect_clip_atlas(self.bundle, ANIM_NAME)
        self.assertTrue(details["atlas_editable"])
        self.assertTrue(details["frame_atlas"])
        self.assertEqual(details["atlas_name"], TEXTURE_NAME)
        self.assertIn("逐帧", details["notice"])
        info = animation.inspect_animation(self.bundle, ANIM_NAME)
        self.assertTrue(info["frame_atlas"])
        self.assertTrue(info["editable"])
        self.assertEqual(info["total"], FRAME_COUNT)
        adapted = animation.read_adapted_replacement(self.bundle, ANIM_NAME, self.gif, fps=30)
        self.assertTrue(adapted["frame_atlas"])
        self.assertEqual(len(adapted["frames"]), FRAME_COUNT)
        seq_out = self.root / "sequence.bundle"
        replaced = animation.replace_bundle_sequence(self.bundle, ANIM_NAME, self.gif, seq_out)
        self.assertTrue(replaced["frame_atlas"])
        self.assertEqual(replaced["frames"], FRAME_COUNT)
        self.assertTrue(seq_out.is_file())

    def test_bc7_output_stays_compressed(self):
        bc7_bundle = build_synthetic_bundle(self.root / "bc7.bundle", bc7=True)
        out = self.root / "bc7-out.bundle"
        frame_atlas.replace_texture_frames(bc7_bundle, TEXTURE_NAME, [self.frame], out)
        data = frame_atlas._texture_of(_load_bundle(out), TEXTURE_NAME).read()
        self.assertEqual(int(data.m_TextureFormat), int(TextureFormat.BC7))
        # BC7 每像素 1 字节；若退化成 RGBA 会变成 4 倍。
        self.assertEqual(len(data.image_data), ATLAS_SIZE[0] * ATLAS_SIZE[1])
        atlas = data.image.convert("RGBA")
        # BC7 是有损压缩，允许每通道 3 级误差。
        for point, expected in (((8, 40), (255, 0, 0, 255)), ((40, 40), (0, 0, 255, 255))):
            actual = atlas.getpixel(point)
            self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(actual, expected)),
                            f"{point}: {actual} != {expected}")

    def test_rejected_input_writes_nothing(self):
        one = self.root / "one"
        one.mkdir()
        Image.new("RGBA", FRAME_SIZE, "red").save(one / "only.png")
        out = self.root / "never.bundle"
        with self.assertRaisesRegex(ValueError, "2 到 512"):
            frame_atlas.replace_animation_frames(self.bundle, ANIM_NAME, one, out)
        self.assertFalse(out.exists())
        small = self.root / "small.png"
        Image.new("RGBA", (20, 20), "red").save(small)
        with self.assertRaisesRegex(ValueError, "整张图集"):
            replace_clip_atlas_checked(self.bundle, ANIM_NAME, small, out)
        self.assertFalse(out.exists())

    def test_inline_stream_skips_textures_already_replaced(self):
        from types import SimpleNamespace

        replaced = SimpleNamespace(
            m_Name="Idle-001", m_StreamData=SimpleNamespace(size=99),
            image_data=b"new", m_CompleteImageSize=3, save=lambda: None,
        )
        other = SimpleNamespace(
            m_Name="Hit-001", m_StreamData=SimpleNamespace(size=99),
            image_data=b"", m_CompleteImageSize=0, save=lambda: None,
            get_image_data=lambda: b"old",
        )
        env = SimpleNamespace(
            objects=[SimpleNamespace(type=SimpleNamespace(name="Texture2D"), read=lambda: replaced),
                     SimpleNamespace(type=SimpleNamespace(name="Texture2D"), read=lambda: other)],
        )
        count = frame_atlas._inline_all_textures(env, exclude={"Idle-001"})
        self.assertEqual(count, 1)
        self.assertEqual(replaced.image_data, b"new")

    def test_compose_grid_writes_dense_mesh_quads(self):
        quads = tuple(
            frame_atlas.Quad(
                (column * 8, row * 8, (column + 1) * 8, (row + 1) * 8),
                (column / 8, row / 8, (column + 1) / 8, (row + 1) / 8),
            )
            for row in range(8) for column in range(8)
        )
        frame = frame_atlas.Frame("grid", 0, (0.0, 0.0, 64.0, 64.0), quads)
        atlas = frame_atlas.compose_grid(
            [frame], [Image.new("RGBA", (64, 64), (255, 0, 0, 255))], (64, 64)
        )
        self.assertEqual(atlas.getpixel((4, 4)), (255, 0, 0, 255))
        self.assertEqual(atlas.getpixel((60, 60)), (255, 0, 0, 255))

    def test_analyze_frames_detects_tile_layout(self):
        frames, tile = frame_atlas.analyze_frames(_load_bundle(self.bundle), TEXTURE_NAME)
        self.assertTrue(tile)
        self.assertEqual(len(frames), FRAME_COUNT)

    def test_command_line_replaces_and_validates(self):
        out = self.root / "cli.bundle"
        root = Path(__file__).resolve().parent
        environment = dict(os.environ, PYTHONPATH=str(root))
        process = subprocess.run(
            [sys.executable, "-m", "astral_party_auto.modkit.frame_atlas",
             "--bundle", str(self.bundle), "--anim", ANIM_NAME,
             "--frames", str(self.folder), "--out", str(out)],
            cwd=root, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=environment,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["frames"], FRAME_COUNT)
        self.assertEqual(payload["texture"], TEXTURE_NAME)
        self.assertEqual(payload["out"], str(out))
        self.assertTrue(out.is_file())
        self.assertEqual(frame_atlas.inspect_texture(out, TEXTURE_NAME)["frames"], FRAME_COUNT)

    def test_fit_frame_keeps_aspect_and_bottom(self):
        source = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        fitted = frame_atlas._fit_frame(source, (40, 40))
        self.assertEqual(fitted.size, (40, 40))
        self.assertEqual(fitted.getchannel("A").getbbox(), (0, 20, 40, 40))

    def test_pick_index_holds_and_subsamples(self):
        self.assertEqual([frame_atlas._pick_index(5, index, 15) for index in range(15)],
                         [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3)
        self.assertEqual([frame_atlas._pick_index(30, index, 15) for index in range(15)],
                         list(range(1, 30, 2)))

    def test_fit_avoids_placeholder_without_shrinking_transparent_area(self):
        safe = (0, 0, 100, 100)
        partially_transparent = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        partially_transparent.paste(Image.new("RGBA", (50, 50), (0, 0, 0, 0)), (50, 0))
        fitted, place_x, place_y = frame_atlas._fit_without_collision(
            partially_transparent, safe, [(50, 0, 100, 50)])
        self.assertEqual(fitted.size, (100, 100))
        self.assertEqual(fitted.getchannel("A").getbbox(), (0, 50, 50, 100))
        self.assertEqual((place_x, place_y), (0, 0))

        opaque = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        fitted, place_x, _place_y = frame_atlas._fit_without_collision(
            opaque, safe, [(50, 0, 100, 100)])
        self.assertLessEqual(place_x + fitted.width, 50)

    def test_frame_source_detection(self):
        self.assertTrue(frame_atlas.is_frame_source(self.gif))
        self.assertTrue(frame_atlas.is_frame_source(self.folder))
        self.assertFalse(frame_atlas.is_frame_source(self.root / "synthetic.bundle"))
        self.assertFalse(frame_atlas.is_frame_source(self.frame))

    def test_gui_candidate_preview_and_draft_save(self):
        api = object.__new__(DesktopApi)
        api.controller = SimpleNamespace(_draft_dir=lambda: self.root / "empty-draft")
        selection = {"name": ANIM_NAME, "bundle": self.bundle.name,
                     "original_path": str(self.bundle), "frame_atlas": True}
        preview = api._clip_candidate_preview(selection, self.gif)
        self.assertEqual(len(preview["frames"]), FRAME_COUNT)
        self.assertTrue(preview["frame_atlas"])

        controller = object.__new__(ModController)
        controller.aa_dir = self.root
        controller.aa_dirs = (self.root,)
        controller.draft_items = []
        controller.log = lambda _message: None
        controller._require_game = lambda: None
        controller._draft_dir = lambda: self.root / "saved-draft"
        controller._save_draft = lambda: None
        controller.bundle_path = lambda _name: self.bundle
        controller._check_sequence_overlap = lambda *args: None
        (self.root / "saved-draft").mkdir()
        item = controller.add_anim_to_draft(self.bundle, ANIM_NAME, frames_path=self.folder,
                                            bundle_name=self.bundle.name)
        self.assertEqual(item["kind"], "texture")
        self.assertEqual(item["name"], TEXTURE_NAME)
        self.assertTrue(item["frame_atlas"])
        draft = self.root / "saved-draft" / self.bundle.name
        self.assertTrue(draft.is_file())
        self.assertEqual(frame_atlas.inspect_texture(draft, TEXTURE_NAME)["frames"], FRAME_COUNT)

        controller.draft_items = [item]
        updated = controller.update_draft_texture(0, self.gif)
        self.assertEqual(updated["name"], TEXTURE_NAME)
        self.assertTrue(updated["frame_atlas"])


def check_real_bundle(source: Path, evidence: Path) -> dict:
    """用真实游戏包做一次隔离验证；源文件只读，写入都在临时目录。"""
    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    evidence.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame-atlas-real-") as folder:
        root = Path(folder)
        original = root / source.name
        original.write_bytes(payload)
        # 先按图集反拼原始 60 帧，再用这些帧逐帧写回：可同时验证 UV 朝向与对齐。
        preview = frame_atlas.read_frames_preview(original, "Idle")
        frames = []
        frames_folder = root / "idle-frames"
        frames_folder.mkdir()
        for index, data in enumerate(preview["frames"]):
            image = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGBA")
            image.save(frames_folder / f"frame_{index:03d}.png")
            frames.append(image)
        out = root / "candidate.bundle"
        start = time.perf_counter()
        result = frame_atlas.replace_animation_frames(original, "Idle", frames_folder, out)
        elapsed = time.perf_counter() - start
        info = frame_atlas.inspect_texture(out, "Idle-001")
        assert result["frames"] == 60 and info["frames"] == 60, (result, info)
        size = out.stat().st_size
        assert size <= 40 * 1024 * 1024, f"单包超过 40MB：{size}"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "源包被修改"
        modified = frame_atlas.read_frames_preview(out, "Idle")
        errors = []
        for before, after in zip(frames, modified["frames"]):
            image = Image.open(io.BytesIO(base64.b64decode(after.split(",", 1)[1]))).convert("RGBA")
            background = Image.new("RGBA", image.size, (60, 65, 76, 255))
            expected = background.copy()
            expected.alpha_composite(before)
            actual = background.copy()
            actual.alpha_composite(image)
            errors.append(max(ImageStat.Stat(ImageChops.difference(
                expected.convert("RGB"), actual.convert("RGB"))).mean))
        max_error = round(max(errors), 3)
        assert max_error < 3, f"逐帧写回后画面偏差过大：{max_error}"
        report = {"bundle": source.name, "frames": result["frames"], "tiles": info["tiles"],
                  "atlas": info["atlas"], "output_bytes": size,
                  "output_mb": round(size / 1024 / 1024, 2),
                  "seconds": round(elapsed, 2), "compressed_bc7": True,
                  "roundtrip_max_mean_error": max_error, "source_unchanged": True}
    (evidence / "frame-atlas-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-bundle", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("qa-evidence/frame-atlas"))
    args = parser.parse_args()
    checks = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FrameAtlasChecks))
    if not checks.wasSuccessful():
        raise SystemExit(1)
    if args.game_bundle:
        print(json.dumps(check_real_bundle(args.game_bundle, args.evidence), ensure_ascii=False))
