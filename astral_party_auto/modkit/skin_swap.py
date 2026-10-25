"""同一角色的两个皮肤互换外观。

保留每个皮肤自己的动作列表、控制器和音效，只交换可见像素：
局内小人按动作逐个渲染另一皮肤的帧，再写回本皮肤图集；立绘交换同名贴图。
生成一个普通 Mod 文件夹，可直接用现有 Mod 管理器安装和还原。
"""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .animation import _load_bundle
from .frame_atlas import (
    _texture_of,
    analyze_frames,
    find_frame_texture,
    render_frame,
    replace_animation_frames_batch,
)
from .hero_skins import SkinInfo, _replace_textures_batch, list_skins, portrait_targets


def _made_dir() -> Path:
    from ..mod_controller import MADE_DIR

    return Path(MADE_DIR)


def _original(controller, bundle: str) -> Path:
    path = controller.original_bundle_path(bundle)
    if path is None:
        path = controller.bundle_path(bundle)
    if path is None:
        raise ValueError(f"找不到资源包：{bundle}")
    return Path(path)


def _render_frames(bundle: Path, action: str, out_dir: Path) -> int:
    env = _load_bundle(bundle)
    texture_name = find_frame_texture(env, action)
    frames, tile = analyze_frames(env, texture_name)
    if not tile:
        raise ValueError(f"{action}: 紧贴网格暂不支持互换")
    data = _texture_of(env, texture_name).read()
    if data.image is None:
        raise ValueError(f"{action}: 图集无法解码")
    atlas = data.image.convert("RGBA")
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        render_frame(frame, atlas).save(out_dir / f"{frame.index:05d}.png")
    atlas.close()
    return len(frames)


def _portrait_key(target) -> tuple:
    return target.category, target.layer, target.variant, target.sfw


def _extract_portrait(bundle: Path, name: str, out_png: Path) -> tuple[int, int]:
    env = _load_bundle(bundle)
    data = _texture_of(env, name).read()
    if data.image is None:
        raise ValueError(f"{name}: 立绘无法解码")
    image = data.image.convert("RGBA")
    try:
        image.save(out_png)
        return image.size
    finally:
        image.close()


def _skin_by_key(controller, hero_id: str, key: str) -> SkinInfo:
    for skin in list_skins(controller, hero_id):
        if skin.key == key:
            return skin
    available = "、".join(skin.key for skin in list_skins(controller, hero_id))
    raise ValueError(f"角色 {hero_id} 没有皮肤 {key}；可用：{available}")


def plan_swap(controller, hero_id: str, skin_a: str, skin_b: str, *, portraits: bool = True) -> dict:
    """返回两个皮肤可互换的动作/立绘，不写任何文件。"""
    a = _skin_by_key(controller, str(hero_id), str(skin_a))
    b = _skin_by_key(controller, str(hero_id), str(skin_b))
    if a.key == b.key:
        raise ValueError("请选择两个不同的皮肤。")
    if a.bundle == b.bundle:
        raise ValueError("两个皮肤使用同一个资源包，无法互换。")
    actions = sorted(set(a.actions) & set(b.actions))
    only_a = sorted(set(a.actions) - set(b.actions))
    only_b = sorted(set(b.actions) - set(a.actions))
    portrait_pairs = []
    portrait_skipped = []
    if portraits:
        targets_a = {_portrait_key(target): target for target in portrait_targets(controller, a)}
        targets_b = {_portrait_key(target): target for target in portrait_targets(controller, b)}
        for key in sorted(set(targets_a) & set(targets_b), key=lambda item: [str(part) for part in item]):
            target_a, target_b = targets_a[key], targets_b[key]
            portrait_pairs.append({
                "key": list(key), "name_a": target_a.name, "name_b": target_b.name,
                "required": target_a.required and target_b.required,
            })
        for key in sorted(set(targets_a) - set(targets_b), key=lambda item: [str(part) for part in item]):
            portrait_skipped.append({"name": targets_a[key].name, "reason": "仅皮肤 A 有"})
        for key in sorted(set(targets_b) - set(targets_a), key=lambda item: [str(part) for part in item]):
            portrait_skipped.append({"name": targets_b[key].name, "reason": "仅皮肤 B 有"})
    if not actions and not portrait_pairs:
        raise ValueError("两个皮肤没有可互换的公共动作或立绘。")
    return {
        "hero_id": a.hero_id,
        "skin_a": {"key": a.key, "label": a.label, "bundle": a.bundle, "actions": len(a.actions)},
        "skin_b": {"key": b.key, "label": b.label, "bundle": b.bundle, "actions": len(b.actions)},
        "actions": actions,
        "only_a": only_a,
        "only_b": only_b,
        "portraits": portrait_pairs,
        "portrait_skipped": portrait_skipped,
        "can_apply": bool(actions or portrait_pairs),
    }


def swap_skins(
    controller,
    hero_id: str,
    skin_a: str,
    skin_b: str,
    *,
    portraits: bool = True,
    install: bool = False,
) -> dict:
    """构建双向互换 Mod；全部成功后才写入 made_mods 并（可选）安装。"""
    plan = plan_swap(controller, hero_id, skin_a, skin_b, portraits=portraits)
    a = _skin_by_key(controller, str(hero_id), str(skin_a))
    b = _skin_by_key(controller, str(hero_id), str(skin_b))
    source_a = _original(controller, a.bundle)
    source_b = _original(controller, b.bundle)
    if source_a.resolve() == source_b.resolve():
        raise ValueError("两个皮肤的源资源包相同，无法互换。")

    stage = _made_dir() / f"_swap_stage_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Path] = {}
    try:
        frames: dict[tuple[str, str], Path] = {}
        common = plan["actions"]
        for action in common:
            frames[("a", action)] = stage / "frames_a" / action
            frames[("b", action)] = stage / "frames_b" / action
            _render_frames(source_a, action, frames[("a", action)])
            _render_frames(source_b, action, frames[("b", action)])
        if common:
            out_a = stage / a.bundle
            out_b = stage / b.bundle
            result_a = replace_animation_frames_batch(
                source_a, {action: frames[("b", action)] for action in common},
                out_a, packer="lzma", inline_stream=True, unbounded_source=True,
            )
            result_b = replace_animation_frames_batch(
                source_b, {action: frames[("a", action)] for action in common},
                out_b, packer="lzma", inline_stream=True, unbounded_source=True,
            )
            outputs[a.bundle] = out_a
            outputs[b.bundle] = out_b
        else:
            result_a = result_b = {"frame_counts": {}}

        replaced_portraits = 0
        if plan["portraits"]:
            targets_a = {target.name: target for target in portrait_targets(controller, a)}
            targets_b = {target.name: target for target in portrait_targets(controller, b)}
            grouped_a: dict[str, dict[str, Path]] = {}
            grouped_b: dict[str, dict[str, Path]] = {}
            for index, pair in enumerate(plan["portraits"]):
                image_a = stage / "portraits" / f"a_{index:03d}.png"
                image_b = stage / "portraits" / f"b_{index:03d}.png"
                image_a.parent.mkdir(parents=True, exist_ok=True)
                target_a, target_b = targets_a[pair["name_a"]], targets_b[pair["name_b"]]
                bundle_b = _original(controller, target_b.bundles[0])
                bundle_a = _original(controller, target_a.bundles[0])
                _extract_portrait(bundle_b, target_b.name, image_b)
                _extract_portrait(bundle_a, target_a.name, image_a)
                for bundle in target_a.bundles:
                    grouped_a.setdefault(bundle, {})[target_a.name] = image_b
                for bundle in target_b.bundles:
                    grouped_b.setdefault(bundle, {})[target_b.name] = image_a
            for bundle, replacements in grouped_a.items():
                target = stage / bundle
                _replace_textures_batch(_original(controller, bundle), replacements, target)
                outputs[bundle] = target
                replaced_portraits += len(replacements)
            for bundle, replacements in grouped_b.items():
                target = stage / bundle
                _replace_textures_batch(_original(controller, bundle), replacements, target)
                outputs[bundle] = target
                replaced_portraits += len(replacements)

        if not outputs:
            raise ValueError("没有生成任何互换资源。")
        pack = _made_dir() / (
            f"皮肤互换_{a.hero_id}_{a.key}_{b.key}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        pack.mkdir(parents=True, exist_ok=False)
        for bundle, source in outputs.items():
            shutil.move(str(source), pack / bundle)
        (pack / "README.txt").write_text(
            "\n".join([
                f"{a.hero_id} · {a.label} ↔ {b.label} 皮肤外观互换",
                "",
                f"动作：{len(common)} 个公共动作双向互换；"
                f"仅 A 有 {len(plan['only_a'])} 个、仅 B 有 {len(plan['only_b'])} 个保持原样。",
                f"立绘：{replaced_portraits} 个贴图双向互换。",
                "安装前退出游戏；安装时管理器会备份原资源包，卸载或一键还原可恢复。",
                "仅替换外观，不改变动作列表、控制器、音效和角色拥有状态。",
            ]) + "\n",
            encoding="utf-8",
        )
        installed = False
        if install:
            controller.install(pack, name=pack.name)
            installed = True
        return {
            "pack": str(pack),
            "name": pack.name,
            "installed": installed,
            "actions": len(common),
            "portraits": replaced_portraits,
            "only_a": plan["only_a"],
            "only_b": plan["only_b"],
            "portrait_skipped": plan["portrait_skipped"],
            "frame_counts": {"a": result_a.get("frame_counts", {}), "b": result_b.get("frame_counts", {})},
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)
