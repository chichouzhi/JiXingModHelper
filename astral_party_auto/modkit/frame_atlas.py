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

import copy
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from UnityPy.classes import Rectf, StreamingInfo, SubMesh
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
MAX_TILES = 200_000
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


@dataclass(frozen=True)
class MeshFrame:
    """紧贴网格帧：没有四边形瓦片，替换时需要重排图集。"""

    name: str
    index: int
    bbox: tuple[float, float, float, float]
    sprite: object
    render_data: object


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
    # 优先选「动作名-数字」的图集，排除 Fight-move-effect-001 这类同前缀特效图。
    preferred = {texture for texture in candidates
                 if re.fullmatch(re.escape(name) + r"-\d+", texture, re.IGNORECASE)}
    if preferred:
        candidates = preferred
    usable = []
    for texture_name in sorted(candidates):
        try:
            if analyze_frames(env, texture_name)[0]:
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
    """把第 index 个目标帧映射到素材帧：多则抽帧，少则每张保持若干帧。"""
    if frame_count <= 0:
        return 0
    return min(image_count - 1, ((2 * index + 1) * image_count) // (2 * frame_count))


def _fit_frame(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    """等比缩放后底部居中放入目标帧，避免把角色拉扁。"""
    width, height = size
    scale = min(width / max(1, source.width), height / max(1, source.height))
    new_width = max(1, min(width, round(source.width * scale)))
    new_height = max(1, min(height, round(source.height * scale)))
    resized = source.resize((new_width, new_height), Image.LANCZOS)
    if (new_width, new_height) == (width, height):
        return resized
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((width - new_width) // 2, height - new_height))
    resized.close()
    return canvas


def _placeholder_tiles(frames: list[Frame], size: tuple[int, int],
                       original: Image.Image) -> set[tuple[int, int, int, int]]:
    """找出复用率最高的透明占位格；角色没覆盖到的普通空白格不应跳过。"""
    from collections import Counter

    width, height = size
    counts: Counter[tuple[int, int, int, int]] = Counter()
    for frame in frames:
        for _rect, (u0, v0, u1, v1) in _tile_rects(frame):
            ax0, ax1 = round(u0 * width), round(u1 * width)
            ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            key = (ax0, ay0, ax1, ay1)
            if _is_blank(original.crop(key)):
                counts[key] += 1
    if not counts:
        return set()
    top = max(counts.values())
    threshold = max(1, len(frames) // 3)
    return {key for key, count in counts.items() if count == top and count >= threshold}


def _safe_bbox(frames: list[Frame], size: tuple[int, int],
               placeholders: set[tuple[int, int, int, int]]) -> tuple[float, float, float, float] | None:
    """可写区域的本地包围盒：排除透明占位格后，角色真正占据的范围。

    Fight-move 这类动作的帧画布比角色宽，左侧大片是占位格；直接按整帧
    缩放会把角色压小并裁掉一半。
    """
    width, height = size
    left = top = float("inf")
    right = bottom = float("-inf")
    for frame in frames:
        for (l, t, r, b), (u0, v0, u1, v1) in _tile_rects(frame):
            ax0, ax1 = round(u0 * width), round(u1 * width)
            ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
            if (ax0, ay0, ax1, ay1) in placeholders:
                continue
            left, top = min(left, l), min(top, t)
            right, bottom = max(right, r), max(bottom, b)
    if left == float("inf"):
        return None
    return (left, top, right, bottom)


def _placeholder_rects(frame: Frame, size: tuple[int, int],
                       placeholders: set[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """当前帧里占位格的像素矩形；不同帧的占位位置可能不同，必须逐帧判断。"""
    width, height = size
    rects: set[tuple[int, int, int, int]] = set()
    for (l, t, r, b), (u0, v0, u1, v1) in _tile_rects(frame):
        ax0, ax1 = round(u0 * width), round(u1 * width)
        ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
        if (ax0, ay0, ax1, ay1) in placeholders:
            rects.add((l, t, r, b))
    return sorted(rects)


def _fit_without_collision(source: Image.Image, safe: tuple[int, int, int, int],
                           blockers: list[tuple[int, int, int, int]]):
    """在不盖到透明占位格的前提下，尽量放大素材（优先居中，其次左/右对齐）。"""
    safe_left, safe_top, safe_right, safe_bottom = safe
    safe_width = max(1, safe_right - safe_left)
    safe_height = max(1, safe_bottom - safe_top)
    scale = 1.0
    while scale >= 0.2:
        width = max(1, round(safe_width * scale))
        height = max(1, round(safe_height * scale))
        fitted = _fit_frame(source, (width, height))
        alpha = fitted.getchannel("A")
        for place_x in (safe_left + (safe_width - width) // 2,
                        safe_left, safe_right - width):
            place_y = safe_bottom - height
            collision = False
            for (l, t, r, b) in blockers:
                ix0, iy0 = max(l, place_x), max(t, place_y)
                ix1, iy1 = min(r, place_x + width), min(b, place_y + height)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                crop = alpha.crop((ix0 - place_x, iy0 - place_y, ix1 - place_x, iy1 - place_y))
                # BC7 的透明区域会有 1~3 级 alpha 噪点，不能当成实体像素。
                if crop.getextrema()[1] > 8:
                    collision = True
                    crop.close()
                    break
                crop.close()
            if not collision:
                alpha.close()
                return fitted, place_x, place_y
        alpha.close()
        fitted.close()
        scale -= 0.04
    return None


def _inline_all_textures(env, exclude: set[str] | None = None) -> int:
    """把仍在 .resS 里的贴图改为内嵌，便于整包重压缩并删除 resS。

    exclude 里的贴图已经由本次替换写入新像素，必须跳过；obj.read() 可能仍
    返回带旧 StreamData 的缓存对象，再内嵌一次会把新图覆盖回原图。
    """
    excluded = set(exclude or ())
    count = 0
    for obj in env.objects:
        if obj.type.name != "Texture2D":
            continue
        data = obj.read()
        if str(getattr(data, "m_Name", "")) in excluded:
            continue
        stream = getattr(data, "m_StreamData", None)
        if not stream or int(getattr(stream, "size", 0) or 0) <= 0:
            continue
        raw = data.get_image_data()
        data.image_data = raw
        data.m_CompleteImageSize = len(raw)
        data.m_StreamData = StreamingInfo(offset=0, size=0, path="")
        data.save()
        count += 1
    return count


def _read_replacement_unbounded(path: str | Path, fps: float):
    """内部互换用：PNG 帧目录不限制总像素，仍逐帧检查尺寸。"""
    from .animation import _fps

    source = Path(path)
    if not source.is_dir():
        return _read_replacement(source, fps)
    files = sorted((file for file in source.iterdir()
                    if file.is_file() and file.suffix.lower() == ".png"))
    if not 2 <= len(files) <= 512:
        raise ValueError("帧目录需要至少 2 张、最多 512 张 PNG 图片。")
    frames = []
    for file in files:
        with Image.open(file) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"帧目录只能放静态 PNG：{file.name}")
            _size_limit(*image.size)
            frames.append(image.convert("RGBA"))
    return frames, [1000 / _fps(fps)] * len(frames), "sequence"


def _drop_unused_stream(env, *, force: bool = False) -> list[str]:
    """全部贴图都改为内嵌像素后，删除不再被引用的 .resS，避免体积翻倍。"""
    files = getattr(getattr(env, "file", None), "files", None)
    if not isinstance(files, dict):
        return []
    used: set[str] = set()
    if not force:
        for obj in env.objects:
            if obj.type.name != "Texture2D":
                continue
            stream = getattr(obj.read(), "m_StreamData", None)
            path = str(getattr(stream, "path", "") or "")
            if path and int(getattr(stream, "size", 0) or 0) > 0:
                used.add(Path(path).name)
    removed = []
    for name in list(files):
        if not isinstance(name, str):
            continue
        if name.lower().endswith(".ress") and (force or Path(name).name not in used):
            del files[name]
            removed.append(name)
    return removed


def _tile_mesh(vertices, uvs) -> bool:
    if not vertices or len(vertices) % 4 or len(vertices) != len(uvs):
        return False
    for start in range(0, len(vertices), 4):
        us = {round(uvs[start + index][0], 6) for index in range(4)}
        vs = {round(uvs[start + index][1], 6) for index in range(4)}
        if len(us) != 2 or len(vs) != 2:
            return False
        pairs = {(round(uvs[start + index][0], 6), round(uvs[start + index][1], 6))
                 for index in range(4)}
        if len(pairs) != 4:
            return False
    return True


def analyze_frames(env, texture_name: str):
    """读取帧；返回 (frames, True) 表示四边形瓦片，False 表示紧贴网格。"""
    frames: list[Frame] = []
    meshes: list[MeshFrame] = []
    tiles = 0
    for obj in env.objects:
        if obj.type.name != "Sprite":
            continue
        sprite = obj.read()
        if getattr(sprite, "m_SpriteAtlas", None) and getattr(sprite.m_SpriteAtlas, "m_PathID", 0):
            raise ValueError(f"{sprite.m_Name}: 使用独立 SpriteAtlas，暂不支持逐帧替换")
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
        ppu = float(getattr(sprite, "m_PixelsToUnits", 0) or PPU_DEFAULT)
        xs = [vertex[0] * ppu for vertex in vertices]
        ys = [vertex[1] * ppu for vertex in vertices]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        index = _frame_index(str(sprite.m_Name))
        if _tile_mesh(vertices, uvs):
            tiles += max(1, len(vertices) // 4)
            if tiles > MAX_TILES:
                raise ValueError(f"图集瓦片过多（>{MAX_TILES}），暂不生成预览")
            quads: list[Quad] = []
            for start in range(0, len(vertices), 4):
                qx = [vertices[start + i][0] * ppu for i in range(4)]
                qy = [vertices[start + i][1] * ppu for i in range(4)]
                qu = [uvs[start + i][0] for i in range(4)]
                qv = [uvs[start + i][1] for i in range(4)]
                quads.append(Quad((min(qx), min(qy), max(qx), max(qy)),
                                  (min(qu), min(qv), max(qu), max(qv))))
            frames.append(Frame(str(sprite.m_Name), index, bbox, tuple(quads)))
        meshes.append(MeshFrame(str(sprite.m_Name), index, bbox, sprite, data))
    if not meshes:
        raise ValueError(f"{texture_name}: 没有找到帧 Sprite")
    meshes.sort(key=lambda frame: frame.index)
    if len(frames) == len(meshes):
        frames.sort(key=lambda frame: frame.index)
        return frames, True
    return meshes, False


def _set_quad(sprite, render_data, local: tuple[float, float, float, float],
              uv: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = local
    u0, v0, u1, v1 = uv
    vertices = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    data = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    data += b"".join(struct.pack("<2f", *uv_value) for uv_value in uvs)
    if not render_data.m_SubMeshes:
        raise ValueError(f"{sprite.m_Name}: 没有子网格，暂不支持重排")
    sub = copy.deepcopy(render_data.m_SubMeshes[0])
    sub.indexCount = 12
    sub.vertexCount = 4
    sub.firstByte = 0
    sub.firstVertex = 0
    sub.baseVertex = 0
    render_data.m_SubMeshes = [sub]
    render_data.m_IndexBuffer = [0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 0, 0]
    render_data.m_VertexData.m_VertexCount = 4
    render_data.m_VertexData.m_DataSize = data
    rect = Rectf(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    render_data.textureRect = rect
    sprite.m_Rect = rect
    sprite.save()


def _pack_entries(entries: list[dict], width: int, height: int) -> float:
    total = sum(max(1, entry["content"].width) * max(1, entry["content"].height)
                for entry in entries)
    capacity = width * height * 0.9
    scale = min(1.0, math.sqrt(capacity / total)) if total > capacity else 1.0
    if scale < 1.0:
        for entry in entries:
            content = entry["content"]
            new_size = (max(1, round(content.width * scale)), max(1, round(content.height * scale)))
            entry["content"] = content.resize(new_size, Image.LANCZOS)
            x0, y0, x1, y1 = entry["local"]
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            entry["local"] = (center_x - new_size[0] / 2, center_y - new_size[1] / 2,
                              center_x + new_size[0] / 2, center_y + new_size[1] / 2)
    padding = 1
    order = sorted(range(len(entries)), key=lambda index: entries[index]["content"].height, reverse=True)
    cursor_x = cursor_y = row_height = 0
    for index in order:
        entry = entries[index]
        tile_width, tile_height = entry["content"].size
        if cursor_x + tile_width > width:
            cursor_x = 0
            cursor_y += row_height + padding
            row_height = 0
        if cursor_y + tile_height > height:
            raise ValueError(
                f"图集 {width}×{height} 放不下 {len(entries)} 帧，请减少帧数或缩小素材"
            )
        entry["packed"] = (cursor_x, cursor_y)
        cursor_x += tile_width + padding
        row_height = max(row_height, tile_height)
    return scale


def _rebake_mesh_frames(env, texture_name: str, frames: list[MeshFrame],
                        images: list[Image.Image], size: tuple[int, int]) -> float:
    width, height = size
    entries = []
    for index, frame in enumerate(frames):
        source = images[_pick_index(len(images), index, len(frames))].convert("RGBA")
        x0, y0, x1, y1 = frame.bbox
        frame_width = max(1, round(x1 - x0))
        frame_height = max(1, round(y1 - y0))
        source = source.resize((frame_width, frame_height), Image.LANCZOS)
        bounds = source.getchannel("A").getbbox()
        if bounds is None:
            content = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            local = (center_x - 0.5, center_y - 0.5, center_x + 0.5, center_y + 0.5)
        else:
            left, top, right, bottom = bounds
            content = source.crop(bounds)
            local = (x0 + left, y1 - bottom, x0 + right, y1 - top)
        entries.append({"frame": frame, "content": content, "local": local})
    scale = _pack_entries(entries, width, height)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for entry in entries:
        tile_width, tile_height = entry["content"].size
        pack_x, pack_y = entry["packed"]
        entry["uv"] = (pack_x / width, 1 - (pack_y + tile_height) / height,
                       (pack_x + tile_width) / width, 1 - pack_y / height)
        atlas.alpha_composite(entry["content"], (pack_x, pack_y))
    for entry in entries:
        _set_quad(entry["frame"].sprite, entry["frame"].render_data, entry["local"], entry["uv"])
    _set_image(_texture_of(env, texture_name).read(), atlas)
    atlas.close()
    return scale


def compose_grid(
    frames: list[Frame],
    images: list[Image.Image],
    size: tuple[int, int],
) -> Image.Image:
    """高密度网格帧（皮肤小人常见）：按每个小四边形原样搬运，不做占位格推断。"""
    if not images:
        raise ValueError("没有可用的替换帧")
    width, height = size
    _size_limit(width, height)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    written: set[tuple[int, int, int, int]] = set()
    for index, frame in enumerate(frames):
        source = images[_pick_index(len(images), index, len(frames))].convert("RGBA")
        x0, y0, x1, y1 = frame.bbox
        fitted = _fit_frame(source, (max(1, round(x1 - x0)), max(1, round(y1 - y0))))
        source.close()
        source = fitted
        for (left, top, right, bottom), (u0, v0, u1, v1) in _tile_rects(frame):
            ax0, ax1 = round(u0 * width), round(u1 * width)
            ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            key = (ax0, ay0, ax1, ay1)
            if key in written:
                continue
            written.add(key)
            tile = source.crop((left, top, right, bottom))
            target = (ax1 - ax0, ay1 - ay0)
            if tile.size != target:
                tile = tile.resize(target, Image.LANCZOS)
            atlas.alpha_composite(tile, (ax0, ay0))
        source.close()
    return atlas


def compose(
    frames: list[Frame],
    images: list[Image.Image],
    size: tuple[int, int],
    original: Image.Image | None = None,
) -> Image.Image:
    """把每帧图片按瓦片布局写回一张同尺寸图集。

    规则：
      - 只跳过复用率最高的透明占位格，避免残影；角色没覆盖到的普通空白格照常写入；
      - 同一个图集区域只写一次，防止多帧互相覆盖；
      - 帧数与图片数不同时均匀取样：素材多则抽帧，素材少则每张保持若干帧；
      - 素材等比缩放、底部居中，不拉伸变形。
    """
    if not images:
        raise ValueError("没有可用的替换帧")
    width, height = size
    _size_limit(width, height)
    placeholders = _placeholder_tiles(frames, size, original) if original is not None else set()
    safe = _safe_bbox(frames, size, placeholders) if placeholders else None
    if safe is None:
        x0, y0, x1, y1 = frames[0].bbox
        safe = (0, 0, x1 - x0, y1 - y0)
    safe = (round(safe[0]), round(safe[1]), round(safe[2]), round(safe[3]))
    fallback_width = max(1, safe[2] - safe[0])
    fallback_height = max(1, safe[3] - safe[1])
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    written: set[tuple[int, int, int, int]] = set()
    for index, frame in enumerate(frames):
        source = images[_pick_index(len(images), index, len(frames))].convert("RGBA")
        blockers = _placeholder_rects(frame, size, placeholders) if placeholders else []
        placed = _fit_without_collision(source, safe, blockers) if blockers else None
        if placed is None:
            fitted = _fit_frame(source, (fallback_width, fallback_height))
            place_x, place_y = safe[0], safe[1]
        else:
            fitted, place_x, place_y = placed
        source.close()
        try:
            for (left, top, right, bottom), (u0, v0, u1, v1) in _tile_rects(frame):
                ax0, ax1 = round(u0 * width), round(u1 * width)
                ay0, ay1 = round((1 - v1) * height), round((1 - v0) * height)
                if ax1 <= ax0 or ay1 <= ay0:
                    continue
                key = (ax0, ay0, ax1, ay1)
                if key in written or key in placeholders:
                    continue
                # 素材边缘可能只覆盖瓦片的一部分：只写交集，避免整块裁掉或拉伸。
                cx0, cy0 = max(left, place_x), max(top, place_y)
                cx1 = min(right, place_x + fitted.width)
                cy1 = min(bottom, place_y + fitted.height)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                written.add(key)
                tile_width = max(1, right - left)
                tile_height = max(1, bottom - top)
                tx0 = round(ax0 + (cx0 - left) / tile_width * (ax1 - ax0))
                tx1 = round(ax0 + (cx1 - left) / tile_width * (ax1 - ax0))
                ty0 = round(ay0 + (cy0 - top) / tile_height * (ay1 - ay0))
                ty1 = round(ay0 + (cy1 - top) / tile_height * (ay1 - ay0))
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                tile = fitted.crop((cx0 - place_x, cy0 - place_y, cx1 - place_x, cy1 - place_y))
                tile = tile.resize((tx1 - tx0, ty1 - ty0), Image.LANCZOS)
                atlas.alpha_composite(tile, (tx0, ty0))
                tile.close()
        finally:
            fitted.close()
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
        adapted = [_fit_frame(images[_pick_index(len(images), index, len(frames))], sizes[index])
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
    result = replace_animation_frames_batch(
        src_bundle, {anim_name: replacement_path}, out_bundle, fps
    )
    texture_name = result["textures"][anim_name]
    return {
        "texture": texture_name,
        "frames": result["frame_counts"][anim_name],
        "atlas": result["atlas_sizes"][texture_name],
        "images": result["source_counts"][anim_name],
        "source_frames": result["source_counts"][anim_name],
        "source_duration": result["source_durations"][anim_name],
        "mesh_rebaked": result["mesh_rebaked"],
        "frame_atlas": True,
    }


@_release_readers
def replace_animation_frames_batch(
    src_bundle: str | Path,
    replacements: dict[str, str | Path],
    out_bundle: str | Path,
    fps: float = 30,
    packer: str = "original",
    inline_stream: bool = False,
    unbounded_source: bool = False,
) -> dict:
    """一次写回同一个资源包里的多个动作，最后统一压缩和回读校验。

    同一角色的全部动作共用一个 controller 资源包，逐个动作保存会反复读写
    几十 MB；这里只加载一次、逐个动作合成，最后原子写一次。
    紧贴网格（非四边形瓦片）的皮肤会自动重排成四边形并重新打包图集。
    """
    if not replacements:
        raise ValueError("没有要替换的动作")
    env = _load_bundle(src_bundle)
    textures: dict[str, str] = {}
    frame_counts: dict[str, int] = {}
    atlas_sizes: dict[str, tuple[int, int]] = {}
    source_counts: dict[str, int] = {}
    source_durations: dict[str, int] = {}
    expected: dict[str, tuple[int, tuple[int, int]]] = {}
    scales: dict[str, float] = {}
    for anim_name, replacement_path in replacements.items():
        texture_name = find_frame_texture(env, anim_name)
        frames, tile = analyze_frames(env, texture_name)
        data = _texture_of(env, texture_name).read()
        size = (int(data.m_Width), int(data.m_Height))
        images, durations, _ = (
            _read_replacement_unbounded(replacement_path, fps)
            if unbounded_source else _read_replacement(replacement_path, fps)
        )
        source_counts[anim_name] = len(images)
        source_durations[anim_name] = sum(durations)
        try:
            if tile:
                if max(len(frame.quads) for frame in frames) > 64:
                    atlas = compose_grid(frames, images, size)
                else:
                    original = data.image.convert("RGBA") if data.image is not None else None
                    try:
                        atlas = compose(frames, images, size, original)
                    finally:
                        if original is not None:
                            original.close()
                _set_image(data, atlas)
                atlas.close()
            else:
                scales[anim_name] = _rebake_mesh_frames(env, texture_name, frames, images, size)
        finally:
            for image in images:
                image.close()
        textures[anim_name] = texture_name
        frame_counts[anim_name] = len(frames)
        atlas_sizes[texture_name] = size
        expected[texture_name] = (len(frames), size)

    def validate(saved) -> None:
        for texture_name, (frame_count, size) in expected.items():
            _validate(saved, texture_name, frame_count, size)

    removed = _drop_unused_stream(env)
    inlined = _inline_all_textures(env, exclude=set(textures.values())) if inline_stream else 0
    if inline_stream and inlined:
        removed += _drop_unused_stream(env, force=True)
    _atomic_save(env, out_bundle, validate, packer=packer)
    return {
        "actions": len(replacements),
        "textures": textures,
        "frame_counts": frame_counts,
        "atlas_sizes": atlas_sizes,
        "source_counts": source_counts,
        "source_durations": source_durations,
        "frames": sum(count for count, _ in expected.values()),
        "mesh_rebaked": sorted(scales),
        "stream_inlined": inlined,
        "stream_removed": removed,
        "packer": packer,
        "frame_atlas": True,
    }


def _validate(env, texture_name: str, frame_count: int, size: tuple[int, int]) -> None:
    saved_frames = read_frames(env, texture_name)
    if len(saved_frames) != frame_count:
        raise ValueError("写回校验失败：帧数发生变化")
    saved_texture = _texture_of(env, texture_name).read()
    if (int(saved_texture.m_Width), int(saved_texture.m_Height)) != size:
        raise ValueError("写回校验失败：图集尺寸发生变化")


def main(argv: list[str] | None = None) -> int:
    """命令行入口：一条命令完成逐帧写回并回读校验。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m astral_party_auto.modkit.frame_atlas",
        description="多瓦片图集逐帧替换：把 GIF/APNG/WebP 或 PNG 帧目录写回动画图集。",
    )
    parser.add_argument("--bundle", required=True, type=Path, help="原始或草稿资源包")
    parser.add_argument("--anim", required=True, help="动画名，例如 Idle、Walk-Back")
    parser.add_argument("--frames", required=True, type=Path, help="GIF/APNG/WebP 文件或 PNG 帧目录")
    parser.add_argument("--out", required=True, type=Path, help="输出的新资源包")
    parser.add_argument("--fps", type=float, default=30.0, help="PNG 帧目录的预览帧率（默认 30）")
    args = parser.parse_args(argv)
    result = replace_animation_frames(args.bundle, args.anim, args.frames, args.out, args.fps)
    result = {**result, "out": str(args.out), "bytes": args.out.stat().st_size}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
