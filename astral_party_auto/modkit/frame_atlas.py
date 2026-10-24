"""多瓦片图集的逐帧替换。

有些游戏的序列帧并不是「一帧一张图」，而是一帧由若干四边形瓦片拼成
（例如 3x3 共 9 块、或 28 块），瓦片在图集里任意摆放，运行时靠网格 UV 拼回。
这类动画的 AnimationClip 常常带变换曲线，现有的单轨解析器读不出来；
而「整张图集替换」又要求用户按碎片布局重绘，不现实。

本模块不解析 AnimationClip：直接读网格四边形，按
「本地矩形 -> 图集 UV 矩形」的对应关系搬运像素，因此不受上述限制。

支持：单个动画对应一张图集、瓦片无旋转/无翻转、瓦片之间不重叠。
不支持：一个动画引用多张图集、独立 alpha 图集、骨骼/蒙皮动画、旋转或翻转的 UV。
遇到这些情况一律报错，不会写出半成品。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from UnityPy.helpers.MeshHelper import MeshHandler

from .animation import (
    _atomic_save,
    _load_bundle,
    _preview,
    _preview_indices,
    _read_replacement,
    _release_readers,
    _set_image,
    _size_limit,
)

PPU_DEFAULT = 100.0
# 原始图集里不透明度低于该比例的瓦片，视为打包器复用的透明占位格。
BLANK_ALPHA_RATIO = 0.01
MAX_TILES = 8192
FRAME_NOTICE = "按图集瓦片的本地位置逐帧写回；游戏内播放速度和顺序不变。"


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


def _tile_rects(frame: Frame) -> list[tuple[tuple[int, int, int, int], tuple[float, float, float, float]]]:
    """把帧的瓦片换算成「帧内像素矩形 -> 图集 UV 矩形」。"""
    x0, y0, x1, y1 = frame.bbox
    frame_width = max(1, round(x1 - x0))
    frame_height = max(1, round(y1 - y0))
    rects = []
    for quad in frame.quads:
        qx0, qy0, qx1, qy1 = quad.local
        left = max(0, round(qx0 - x0))
        right = min(frame_width, round(qx1 - x0))
        top = max(0, round(y1 - qy1))
        bottom = min(frame_height, round(y1 - qy0))
        if right <= left or bottom <= top:
            continue
        rects.append(((left, top, right, bottom), quad.uv))
    return rects


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


def _texture_names(env) -> set[str]:
    return {
        str(obj.peek_name() or "")
        for obj in env.objects
        if obj.type.name == "Texture2D" and obj.peek_name()
    }


def find_frame_texture(env, asset_name: str) -> str:
    """从动画名/贴图名找到唯一的帧图集。

    依次尝试：同名贴图、`动画名_帧号` 的 Sprite 前缀、`动画名-001` 贴图。
    找到多张或一张都找不到时明确报错，不猜第一张。
    """
    name = str(asset_name or "")
    if not name:
        raise ValueError("动画名称为空，无法定位帧图集")
    local = _texture_names(env)
    sprite_candidates: set[str] = set()
    prefix = name.casefold() + "_"
    for obj in env.objects:
        if obj.type.name != "Sprite":
            continue
        if not str(obj.peek_name() or "").casefold().startswith(prefix):
            continue
        sprite = obj.read()
        pointer = sprite.m_RD.texture
        reader = pointer.deref() if pointer is not None else None
        if reader is None:
            continue
        texture_name = str(reader.read().m_Name or "")
        if texture_name in local:
            sprite_candidates.add(texture_name)
    if sprite_candidates:
        candidates = sprite_candidates
    elif name in local:
        candidates = {name}
    else:
        lowered = name.casefold()
        candidates = {texture_name for texture_name in local
                      if texture_name.casefold().startswith(lowered + "-")}
    if not candidates:
        raise ValueError(f"{name}: 没有找到可逐帧替换的图集，此动画可能引用外部图集或骨骼网格")
    usable = []
    for texture_name in sorted(candidates):
        try:
            if read_frames(env, texture_name):
                usable.append(texture_name)
        except ValueError:
            continue
    if not usable:
        raise ValueError(f"{name}: 候选图集里没有帧网格，暂不支持逐帧替换")
    if len(usable) > 1:
        raise ValueError(
            f"{name}: 找到 {len(usable)} 张帧图集（{', '.join(usable)}），"
            "一个动画只能对应一张图集，暂不支持逐帧替换"
        )
    return usable[0]


def _texture_of(env, texture_name: str):
    for obj in env.objects:
        if obj.type.name == "Texture2D" and (obj.peek_name() or "") == texture_name:
            return obj
    raise ValueError(f"{texture_name}: 没有找到同名贴图")


def atlas_details(env, asset_name: str) -> tuple[object, dict]:
    """解析帧图集并返回 (贴图数据, 供制作页使用的详情)。"""
    texture_name = find_frame_texture(env, asset_name)
    frames = read_frames(env, texture_name)
    data = _texture_of(env, texture_name).read()
    sizes = sorted({(round(f.bbox[2] - f.bbox[0]), round(f.bbox[3] - f.bbox[1])) for f in frames})
    x0, y0, x1, y1 = frames[0].bbox
    return data, {
        "atlas_name": texture_name,
        "atlas_width": int(data.m_Width),
        "atlas_height": int(data.m_Height),
        "atlas_editable": True,
        "atlas_reason": "",
        "frame_atlas": True,
        "frames": len(frames),
        "tiles": sum(len(frame.quads) for frame in frames),
        "frame_sizes": sizes,
        "first_frame_size": (round(x1 - x0), round(y1 - y0)),
        "notice": FRAME_NOTICE,
    }


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


@_release_readers
def inspect_animation(bundle_path: str | Path, anim_name: str) -> dict:
    """按动画名解析帧图集；失败时抛出可读原因。"""
    _, details = atlas_details(_load_bundle(bundle_path), anim_name)
    return details


def _is_blank(tile: Image.Image) -> bool:
    """原始图集里这块区域是不是空白（打包器复用的占位格）。"""
    alpha = tile.getchannel("A")
    histogram = alpha.histogram()
    visible = sum(histogram[16:]) / max(1, tile.width * tile.height)
    return visible < BLANK_ALPHA_RATIO


def _pick_index(image_count: int, index: int, frame_count: int) -> int:
    """帧数与图片数不同时按顺序均匀取样；图片不足时循环使用。"""
    if image_count >= frame_count:
        return min(image_count - 1, round(index * (image_count - 1) / max(1, frame_count - 1)))
    return index % image_count


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
      - 帧数与图片数不同时按顺序均匀取样，图片不足时循环使用。
    """
    if not images:
        raise ValueError("没有可用的替换帧")
    width, height = size
    _size_limit(width, height)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    written: set[tuple[int, int, int, int]] = set()
    for index, frame in enumerate(frames):
        source = images[_pick_index(len(images), index, len(frames))].convert("RGBA")
        x0, y0, x1, y1 = frame.bbox
        frame_width = max(1, round(x1 - x0))
        frame_height = max(1, round(y1 - y0))
        source = source.resize((frame_width, frame_height), Image.LANCZOS)
        for (left, top, right, bottom), (u0, v0, u1, v1) in _tile_rects(frame):
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


def render_frame(frame: Frame, atlas: Image.Image) -> Image.Image:
    """按瓦片布局把图集里的一帧拼回完整画面。"""
    x0, y0, x1, y1 = frame.bbox
    width = max(1, round(x1 - x0))
    height = max(1, round(y1 - y0))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for (left, top, right, bottom), (u0, v0, u1, v1) in _tile_rects(frame):
        ax0, ax1 = round(u0 * atlas.width), round(u1 * atlas.width)
        ay0, ay1 = round((1 - v1) * atlas.height), round((1 - v0) * atlas.height)
        if ax1 <= ax0 or ay1 <= ay0:
            continue
        tile = atlas.crop((ax0, ay0, ax1, ay1))
        tile = tile.resize((right - left, bottom - top), Image.LANCZOS)
        canvas.alpha_composite(tile, (left, top))
    return canvas


@_release_readers
def read_frames_preview(bundle_path: str | Path, anim_name: str, fps: float = 30) -> dict:
    """逐帧预览：从图集反拼每一帧，不依赖 AnimationClip。"""
    env = _load_bundle(bundle_path)
    texture_name = find_frame_texture(env, anim_name)
    frames = read_frames(env, texture_name)
    data = _texture_of(env, texture_name).read()
    atlas = data.image.convert("RGBA") if data.image is not None else Image.new(
        "RGBA", (int(data.m_Width), int(data.m_Height)), (0, 0, 0, 0)
    )
    x0, y0, x1, y1 = frames[0].bbox
    width = max(1, round(x1 - x0))
    height = max(1, round(y1 - y0))
    rate = max(1.0, min(120.0, float(fps)))
    durations = [1000 / rate] * len(frames)
    indices, sampled = _preview_indices(len(frames), durations)
    info = {
        "animation_kind": "sprite_clip",
        "playable": True,
        "editable": True,
        "reason": "",
        "notice": FRAME_NOTICE,
        "frame_atlas": True,
        "texture": texture_name,
        "width": width,
        "height": height,
    }
    return _preview((render_frame(frames[index], atlas) for index in indices),
                    [frames[index].name for index in indices], sampled, len(frames), info=info)


@_release_readers
def read_adapted_frames(bundle_path: str | Path, anim_name: str,
                        replacement_path: str | Path, fps: float = 30) -> dict:
    """预览导入的帧素材写入后的效果（按目标帧数均匀取样、适配帧尺寸）。"""
    env = _load_bundle(bundle_path)
    texture_name = find_frame_texture(env, anim_name)
    frames = read_frames(env, texture_name)
    sizes = [(max(1, round(f.bbox[2] - f.bbox[0])), max(1, round(f.bbox[3] - f.bbox[1])))
             for f in frames]
    rate = max(1.0, min(120.0, float(fps)))
    images, durations, _ = _read_replacement(replacement_path, rate)
    try:
        indices, sampled = _preview_indices(len(frames), [1000 / rate] * len(frames))
        adapted = [images[_pick_index(len(images), index, len(frames))].resize(sizes[index], Image.LANCZOS)
                   for index in indices]
        info = {
            "animation_kind": "sprite_clip", "playable": True, "editable": True,
            "notice": f"按目标的 {len(frames)} 帧写回，帧数不同时均匀取样；游戏内播放时序不变。",
            "frame_atlas": True, "texture": texture_name,
            "width": sizes[0][0], "height": sizes[0][1],
            "source_frame_count": len(images), "source_duration": sum(durations),
        }
        return _preview(adapted, [frames[index].name for index in indices], sampled, len(frames), info=info)
    finally:
        for image in images:
            image.close()


def is_frame_source(path: str | Path) -> bool:
    """判断导入目标是不是「多帧素材」：PNG 帧目录，或 GIF/APNG/动态 WebP。"""
    source = Path(path)
    if source.is_dir():
        return True
    if source.suffix.lower() not in {".gif", ".webp", ".png", ".apng"}:
        return False
    try:
        with Image.open(source) as image:
            return int(getattr(image, "n_frames", 1)) > 1
    except OSError:
        return False


def _load_images(image_paths: list[str | Path]) -> list[Image.Image]:
    """逐张打开静态图片；调用方负责关闭。"""
    images = []
    try:
        for path in image_paths:
            with Image.open(path) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("逐帧替换的每一帧必须是静态图片")
                if Path(path).stat().st_size > 64 * 1024 * 1024:
                    raise ValueError("单帧图片超过 64 MB，请先缩小尺寸")
                images.append(opened.convert("RGBA"))
    except Exception:
        for image in images:
            image.close()
        raise
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
        if original is not None:
            original.close()
    _set_image(data, atlas)
    atlas.close()
    _atomic_save(env, out_bundle, lambda saved: _validate(saved, texture_name, len(frames), size))
    return {
        "texture": texture_name,
        "frames": len(frames),
        "atlas": size,
        "images": len(image_paths),
    }


@_release_readers
def replace_animation_frames(
    src_bundle: str | Path,
    anim_name: str,
    replacement_path: str | Path,
    out_bundle: str | Path,
    fps: float = 30,
) -> dict:
    """按动画名找到帧图集，用 GIF/APNG/WebP 或 PNG 帧目录逐帧写回。"""
    env = _load_bundle(src_bundle)
    texture_name = find_frame_texture(env, anim_name)
    frames = read_frames(env, texture_name)
    data = _texture_of(env, texture_name).read()
    size = (int(data.m_Width), int(data.m_Height))
    original = data.image.convert("RGBA") if data.image is not None else None
    images, durations, _ = _read_replacement(replacement_path, fps)
    try:
        atlas = compose(frames, images, size, original)
    finally:
        for image in images:
            image.close()
        if original is not None:
            original.close()
    _set_image(data, atlas)
    atlas.close()
    _atomic_save(env, out_bundle, lambda saved: _validate(saved, texture_name, len(frames), size))
    return {
        "texture": texture_name,
        "frames": len(frames),
        "atlas": size,
        "images": len(images),
        "source_frames": len(images),
        "source_duration": sum(durations),
        "frame_atlas": True,
    }


def _validate(env, texture_name: str, frame_count: int, size: tuple[int, int]) -> None:
    saved_frames = read_frames(env, texture_name)
    if len(saved_frames) != frame_count:
        raise ValueError("写回校验失败：帧数发生变化")
    saved_texture = _texture_of(env, texture_name).read()
    if (int(saved_texture.m_Width), int(saved_texture.m_Height)) != size:
        raise ValueError("写回校验失败：图集尺寸发生变化")
