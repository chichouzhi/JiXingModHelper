"""多瓦片图集的逐帧替换。

有些游戏的序列帧并不是「一帧一张图」，而是一帧由若干四边形瓦片拼成
（例如 3x3 共 9 块、或 28 块），瓦片在图集里任意摆放，运行时靠网格 UV 拼回。
这类动画的 AnimationClip 常常带变换曲线，现有的单轨解析器读不出来；
而「整张图集替换」又要求用户按碎片布局重绘，不现实。

本模块不解析 AnimationClip：直接读网格四边形，按
「本地矩形 -> 图集 UV 矩形」的对应关系搬运像素，因此不受上述限制。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

from .animation import MAX_FILE_BYTES, _atomic_save, _load_bundle, _set_image, _size_limit

PPU_DEFAULT = 100.0
# 原始图集里不透明度低于该比例的瓦片，视为打包器复用的透明占位格。
BLANK_ALPHA_RATIO = 0.01
MAX_TILES = 8192


@dataclass(frozen=True)
class Quad:
    """一个瓦片：本地像素矩形 + 图集 UV 矩形（V 轴向上）。"""

    local: tuple[float, float, float, float]
    uv: tuple[float, float, float, float]


@dataclass(frozen=True)
class Frame:
    """一帧：整帧本地包围盒 + 组成它的瓦片。"""

    name: str
    index: int
    bbox: tuple[float, float, float, float]
    quads: tuple[Quad, ...]


def _frame_index(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits[-6:] or 0)


def read_frames(env, texture_name: str) -> list[Frame]:
    """取出某个贴图上的所有帧 Sprite 及其瓦片映射，按帧号排序。"""
    frames: list[Frame] = []
    tiles = 0
    for obj in env.objects:
        if obj.type.name != "Sprite":
            continue
        sprite = obj.read()
        data = sprite.m_RD
        pointer = data.texture
        reader = pointer.deref() if pointer is not None else None
        if reader is None or reader.read().m_Name != texture_name:
            continue
        mesh = MeshHandler(data, obj.version)
        mesh.process()
        vertices = mesh.m_Vertices or []
        uvs = mesh.m_UV0 or []
        if not vertices or len(vertices) != len(uvs):
            raise ValueError(f"{sprite.m_Name}: 网格数据缺失，无法读取瓦片")
        if len(vertices) % 4:
            raise ValueError(f"{sprite.m_Name}: 网格顶点数不是 4 的倍数，暂不支持")
        tiles += len(vertices) // 4
        if tiles > MAX_TILES:
            raise ValueError(f"图集瓦片过多（>{MAX_TILES}），暂不生成预览")
        ppu = float(getattr(sprite, "m_PixelsToUnits", 0) or PPU_DEFAULT)
        xs = [v[0] * ppu for v in vertices]
        ys = [v[1] * ppu for v in vertices]
        quads: list[Quad] = []
        for start in range(0, len(vertices), 4):
            qx = [vertices[start + i][0] * ppu for i in range(4)]
            qy = [vertices[start + i][1] * ppu for i in range(4)]
            qu = [uvs[start + i][0] for i in range(4)]
            qv = [uvs[start + i][1] for i in range(4)]
            quads.append(
                Quad(
                    (min(qx), min(qy), max(qx), max(qy)),
                    (min(qu), min(qv), max(qu), max(qv)),
                )
            )
        frames.append(
            Frame(
                str(sprite.m_Name),
                _frame_index(str(sprite.m_Name)),
                (min(xs), min(ys), max(xs), max(ys)),
                tuple(quads),
            )
        )
    frames.sort(key=lambda frame: frame.index)
    if not frames:
        raise ValueError(f"{texture_name}: 没有找到帧 Sprite")
    return frames


def _is_blank(tile: Image.Image) -> bool:
    """原始图集里这块区域是不是空白（打包器复用的占位格）。"""
    alpha = tile.getchannel("A")
    histogram = alpha.histogram()
    visible = sum(histogram[16:]) / max(1, tile.width * tile.height)
    return visible < BLANK_ALPHA_RATIO


def compose(
    frames: list[Frame],
    images: list[Image.Image],
    size: tuple[int, int],
    original: Image.Image | None = None,
) -> Image.Image:
    """把每帧图片按瓦片布局写回一张同尺寸图集。

    规则：
      - 原始图集里是空白的瓦片直接跳过，避免在动画里留下残影块；
      - 同一个图集区域只写一次，防止多帧互相覆盖；
      - 帧数比图片多时循环使用，比图片少时只取前 N 张。
    """
    if not images:
        raise ValueError("没有可用的替换帧")
    width, height = size
    _size_limit(width, height)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    written: set[tuple[int, int, int, int]] = set()
    for index, frame in enumerate(frames):
        source = images[index % len(images)].convert("RGBA")
        x0, y0, x1, y1 = frame.bbox
        frame_width = max(1, round(x1 - x0))
        frame_height = max(1, round(y1 - y0))
        source = source.resize((frame_width, frame_height), Image.LANCZOS)
        for quad in frame.quads:
            qx0, qy0, qx1, qy1 = quad.local
            left = max(0, round(qx0 - x0))
            right = min(frame_width, round(qx1 - x0))
            top = max(0, round(y1 - qy1))
            bottom = min(frame_height, round(y1 - qy0))
            if right <= left or bottom <= top:
                continue
            u0, v0, u1, v1 = quad.uv
            ax0, ax1 = round(u0 * width), round(u1 * width)
            ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            key = (ax0, ay0, ax1, ay1)
            if key in written:
                continue
            written.add(key)
            if original is not None and _is_blank(original.crop(key)):
                continue
            tile = source.crop((left, top, right, bottom))
            tile = tile.resize((ax1 - ax0, ay1 - ay0), Image.LANCZOS)
            atlas.alpha_composite(tile, (ax0, ay0))
    return atlas


def inspect_texture(bundle_path: str | Path, texture_name: str) -> dict:
    """给预览用的元信息：帧数、帧尺寸、瓦片数、是否有透明占位。"""
    env = _load_bundle(bundle_path)
    frames = read_frames(env, texture_name)
    texture = _texture_of(env, texture_name)
    data = texture.read()
    x0, y0, x1, y1 = frames[0].bbox
    sizes = sorted({(round(f.bbox[2] - f.bbox[0]), round(f.bbox[3] - f.bbox[1])) for f in frames})
    return {
        "texture": texture_name,
        "frames": len(frames),
        "frame_sizes": sizes,
        "tiles": sum(len(frame.quads) for frame in frames),
        "atlas": (int(data.m_Width), int(data.m_Height)),
        "first_frame_size": (round(x1 - x0), round(y1 - y0)),
    }


def _texture_of(env, texture_name: str):
    for obj in env.objects:
        if obj.type.name == "Texture2D" and (obj.peek_name() or "") == texture_name:
            return obj
    raise ValueError(f"{texture_name}: 没有找到同名贴图")


def _load_images(image_paths: list[str | Path]) -> list[Image.Image]:
    images = [Image.open(path) for path in image_paths]
    for image in images:
        if getattr(image, "n_frames", 1) == 1 and image.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
            for opened in images:
                opened.close()
            raise ValueError("逐帧替换只接受 PNG、JPG、WebP、BMP 静态图片")
        if Path(image.filename or "").stat().st_size > MAX_FILE_BYTES:
            for opened in images:
                opened.close()
            raise ValueError("单帧图片超过 64 MB，请先缩小尺寸")
    return images


def replace_texture_frames(
    src_bundle: str | Path,
    texture_name: str,
    image_paths: list[str | Path],
    out_bundle: str | Path,
) -> dict:
    """用一组帧图片重建指定贴图，写回新 bundle 并做回读校验。"""
    env = _load_bundle(src_bundle)
    frames = read_frames(env, texture_name)
    texture = _texture_of(env, texture_name)
    data = texture.read()
    size = (int(data.m_Width), int(data.m_Height))
    original = data.image.convert("RGBA") if data.image is not None else None
    images = _load_images(image_paths)
    try:
        atlas = compose(frames, images, size, original)
    finally:
        for image in images:
            image.close()
    _set_image(data, atlas)
    atlas.close()

    def validate(saved) -> None:
        saved_frames = read_frames(saved, texture_name)
        saved_texture = _texture_of(saved, texture_name).read()
        if len(saved_frames) != len(frames):
            raise ValueError("写回校验失败：帧数发生变化")
        if (int(saved_texture.m_Width), int(saved_texture.m_Height)) != size:
            raise ValueError("写回校验失败：图集尺寸发生变化")

    _atomic_save(env, out_bundle, validate)
    return {
        "texture": texture_name,
        "frames": len(frames),
        "atlas": size,
        "images": len(image_paths),
    }
