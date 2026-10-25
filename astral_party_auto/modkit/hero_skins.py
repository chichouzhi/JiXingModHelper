"""整角色 / 全皮肤批量替换。

流程：扫描 AnimatorController 名找到角色拥有的皮肤 → 读取该皮肤的
全部动作与立绘 → 按素材目录匹配 → 在临时目录构建所有资源包 →
全部成功后再一次性写入作品集。任何一步失败都不会改动现有作品集。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import UnityPy
from PIL import Image

from .animation import _atomic_save, _load_bundle, _set_image
from .frame_atlas import _texture_of, replace_animation_frames_batch

CONTROLLER_RE = re.compile(r"^Hero(\d+)(?:_(\d{2}|Max))?_Controller$")
PORTRAIT_CATEGORIES = ("Bust", "Card2", "Card", "LevelUp", "ProfilePhoto", "RolePhoto", "ThinCard")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
SKIN_ORDER = {"base": 0, "Max": 99}


@dataclass(frozen=True)
class SkinInfo:
    hero_id: str
    key: str
    label: str
    controller: str
    bundle: str
    path: Path
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PortraitTarget:
    name: str
    category: str
    layer: int | None
    variant: int | None
    sfw: bool
    required: bool
    bundles: tuple[str, ...]


def _data_dir() -> Path:
    from ..mod_controller import DATA_DIR

    return Path(DATA_DIR)


def _cache_path() -> Path:
    return _data_dir() / "hero_controllers.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _is_under(path: Path, directory: Path) -> bool:
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except (OSError, ValueError):
        return False


def _source_of(controller, path: Path) -> str:
    directories = tuple(getattr(controller, "aa_dirs", ()) or ())
    if len(directories) > 1 and _is_under(path, Path(directories[0])):
        return "cache"
    return "base"


def _cached_controllers(controller) -> dict[str, list[dict]]:
    source_key = str(getattr(controller, "_index_source_key", ""))
    cache = _cache_path()
    if not cache.is_file():
        return {}
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("source_key") != source_key or not isinstance(data.get("controllers"), dict):
        return {}
    return data["controllers"]


def _hero_ids(found: dict[str, list[dict]]) -> list[str]:
    heroes = set()
    for name in found:
        match = CONTROLLER_RE.match(name)
        if match:
            heroes.add(match.group(1))
    return sorted(heroes, key=int)


def controller_summary(controller) -> dict:
    """不触发扫描，返回缓存里已有的角色/皮肤概况。"""
    found = _cached_controllers(controller)
    return {"scanned": bool(found), "heroes": _hero_ids(found), "controllers": len(found)}


def scan_controllers(
    controller,
    *,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[dict]]:
    """扫描带有完整动作的 bundle，按 AnimatorController 名归类。

    结果缓存到 modkit_data/hero_controllers.json；索引来源不变时直接复用。
    一次扫描约 30 秒，会覆盖所有角色，之后按角色查询是即时的。
    """
    source_key = str(getattr(controller, "_index_source_key", ""))
    cache = _cache_path()
    if not refresh:
        cached = _cached_controllers(controller)
        if cached:
            return cached

    anim_index = controller.typed_index.get("anim") or {}
    if not anim_index:
        raise ValueError("动画索引为空，请先在「浏览资源」刷新索引。")
    candidates = [
        (bundle, names)
        for bundle, names in anim_index.items()
        if "Idle" in names and len(names) >= 10
    ]
    if not candidates:
        raise ValueError("索引里没有找到带完整动作的角色资源包。")

    found: dict[str, list[dict]] = {}
    total = len(candidates)
    for position, (bundle, _names) in enumerate(candidates, 1):
        path = controller.bundle_path(bundle)
        if path is None:
            continue
        try:
            env = UnityPy.load(Path(path).read_bytes(), path=str(Path(path).parent))
        except Exception:
            continue
        for obj in env.objects:
            if obj.type.name != "AnimatorController":
                continue
            try:
                name = str(obj.peek_name() or "")
            except Exception:
                continue
            if not CONTROLLER_RE.match(name):
                continue
            found.setdefault(name, []).append(
                {"bundle": bundle, "path": str(path), "source": _source_of(controller, Path(path))}
            )
        if progress is not None and position % 10 == 0:
            progress(position, total)
    if progress is not None:
        progress(total, total)
    if not found:
        raise ValueError("没有找到任何 Hero*_Controller，请确认当前游戏版本。")

    _atomic_write(
        cache,
        json.dumps(
            {"source_key": source_key, "created": datetime.now().isoformat(timespec="seconds"),
             "controllers": found},
            ensure_ascii=False,
            indent=1,
        ),
    )
    return found


def _natural_skin_key(key: str) -> tuple[int, int | str]:
    if key == "base":
        return (0, 0)
    if key == "Max":
        return (2, 0)
    return (1, int(key))


def _skin_label(key: str) -> str:
    return {"base": "基础", "Max": "Max"}.get(key, f"皮肤 {key}")


def list_skins(
    controller,
    hero_id: str,
    *,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[SkinInfo]:
    """列出某个角色在游戏里实际拥有的皮肤。"""
    hero_id = str(hero_id).strip()
    if not hero_id.isdigit():
        raise ValueError("角色 ID 需要是数字，例如 101。")
    found = scan_controllers(controller, refresh=refresh, progress=progress)
    skins: list[SkinInfo] = []
    for name in sorted(found):
        match = CONTROLLER_RE.match(name)
        if not match or match.group(1) != hero_id:
            continue
        entries = found[name]
        cache_entries = [row for row in entries if row.get("source") == "cache"]
        entry = (cache_entries or entries)[0]
        bundle = str(entry["bundle"])
        actions = tuple(sorted(str(action) for action in (controller.typed_index.get("anim") or {}).get(bundle, ())))
        if not actions:
            continue
        skins.append(
            SkinInfo(
                hero_id=hero_id,
                key=match.group(2) or "base",
                label=_skin_label(match.group(2) or "base"),
                controller=name,
                bundle=bundle,
                path=Path(entry["path"]),
                actions=actions,
            )
        )
    skins.sort(key=lambda skin: _natural_skin_key(skin.key))
    return skins


def parse_portrait(name: str, hero_id: str):
    """解析 UT_Hero_* 名称，返回 (皮肤, 类别, Bust 层, ProfilePhoto 变体, sfw)。"""
    prefix = "UT_Hero_"
    if not name.startswith(prefix):
        return None
    body = name[len(prefix):]
    for category in PORTRAIT_CATEGORIES:
        head = f"{category}_{hero_id}"
        if body == head:
            rest = ""
        elif body.startswith(head + "_"):
            rest = body[len(head) + 1:]
        elif (category == "ProfilePhoto" and body.startswith(head)
              and len(body) == len(head) + 1 and body[-1].isdigit()):
            # ProfilePhoto_1010 … 1019 是基础皮肤的好感等级头像。
            rest = body[-1]
        else:
            continue
        parts = [part for part in rest.split("_") if part]
        skin = "base"
        if parts and (parts[0] == "Max" or (len(parts[0]) == 2 and parts[0].isdigit())):
            skin = parts.pop(0)
        sfw = False
        if parts and parts[-1].lower() == "sfw":
            sfw = True
            parts.pop()
        layer = None
        variant = None
        if category == "Bust":
            if parts:
                if len(parts) != 1 or parts[0] not in ("0", "1"):
                    return None
                layer = int(parts.pop())
        elif category == "ProfilePhoto" and skin == "base" and parts:
            if len(parts) != 1 or not parts[0].isdigit():
                return None
            variant = int(parts.pop())
        if parts:
            return None
        return skin, category, layer, variant, sfw
    return None


def portrait_targets(controller, skin: SkinInfo) -> list[PortraitTarget]:
    """列出某皮肤的全部立绘；核心 8 张为必需，sfw / 等级头像为可选。"""
    texture_index = controller.typed_index.get("texture") or {}
    by_name: dict[str, list[str]] = {}
    for bundle, names in texture_index.items():
        for name in names:
            by_name.setdefault(str(name), []).append(bundle)
    targets: list[PortraitTarget] = []
    for name, bundle_names in by_name.items():
        parsed = parse_portrait(name, skin.hero_id)
        if parsed is None:
            continue
        target_skin, category, layer, variant, sfw = parsed
        if target_skin != skin.key:
            continue
        bundles = tuple(
            bundle for bundle in bundle_names if controller.bundle_path(bundle) is not None
        )
        if not bundles:
            continue
        required = not sfw and variant is None
        targets.append(
            PortraitTarget(
                name=name,
                category=category,
                layer=layer,
                variant=variant,
                sfw=sfw,
                required=required,
                bundles=bundles,
            )
        )
    targets.sort(key=lambda target: (
        PORTRAIT_CATEGORIES.index(target.category),
        target.layer if target.layer is not None else -1,
        target.variant if target.variant is not None else -1,
        target.name,
    ))
    return targets


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.casefold())


def _skin_folder_names(skin: SkinInfo) -> set[str]:
    if skin.key == "base":
        return {"base", "default", skin.hero_id, "基础"}
    if skin.key == "Max":
        return {"max", f"{skin.hero_id}_max", f"{skin.hero_id}max"}
    return {
        skin.key,
        f"_{skin.key}",
        f"{skin.hero_id}_{skin.key}",
        f"skin{skin.key}",
        f"皮肤{skin.key}",
    }


def find_skin_dir(root: str | Path, skin: SkinInfo, *, single: bool = False) -> Path | None:
    """在素材根目录里找该皮肤的目录。"""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"素材目录不存在：{root}")
    names = {name.casefold() for name in _skin_folder_names(skin)}
    candidates: list[Path] = []
    hero_dir = root / skin.hero_id
    if hero_dir.is_dir():
        # 优先具体的皮肤子目录，避免把 101/ 误当成 base 而索引到所有皮肤。
        candidates.extend(path for path in hero_dir.iterdir() if path.is_dir())
        candidates.append(hero_dir)
    candidates.append(root)
    candidates.extend(path for path in root.iterdir() if path.is_dir())
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.name.casefold() in names:
            return path
    if single and any((root / sub).is_dir() for sub in ("portraits", "actions")):
        return root
    return None


def index_assets(root: str | Path) -> dict[str, dict[str, list[Path]]]:
    """递归索引图片文件与包含 PNG 的帧目录，供动作和立绘匹配。"""
    root = Path(root)
    files: dict[str, list[Path]] = {}
    folders: dict[str, list[Path]] = {}

    def add(table: dict[str, list[Path]], key: str, path: Path) -> None:
        table.setdefault(key.casefold(), []).append(path)
        normalized = _normalize(key)
        if normalized and normalized != key.casefold():
            table.setdefault(normalized, []).append(path)

    for path in root.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            add(files, path.stem, path)
        elif path.is_dir():
            try:
                has_png = any(child.suffix.lower() == ".png" for child in path.iterdir() if child.is_file())
            except OSError:
                has_png = False
            if has_png:
                add(folders, path.name, path)
    return {"files": files, "folders": folders}


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _resolve(index: dict, aliases: list[str], *, folders: bool = False) -> Path:
    table = index["folders" if folders else "files"]
    for alias in aliases:
        matches = _unique_paths(table.get(alias.casefold(), []))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"素材名 {alias} 匹配到多个文件：{', '.join(str(path) for path in matches)}")
    for alias in aliases:
        matches = _unique_paths(table.get(_normalize(alias), []))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"素材名 {alias} 匹配到多个文件：{', '.join(str(path) for path in matches)}")
    raise FileNotFoundError(aliases[0])


def _portrait_aliases(target: PortraitTarget) -> list[str]:
    aliases = [target.name, f"{target.category}_{target.layer}" if target.layer is not None else target.category]
    if target.layer is not None:
        aliases += [f"{target.category}{target.layer}"]
    if target.sfw:
        aliases += [f"{target.category}_sfw", f"{target.category}sfw"]
    if target.variant is not None:
        aliases += [f"{target.category}_{target.variant}", f"{target.category}{target.variant}"]
    return aliases


def _match(index: dict, aliases: list[str], *, folders: bool = False) -> tuple[Path | None, str]:
    try:
        return _resolve(index, aliases, folders=folders), ""
    except FileNotFoundError:
        return None, ""
    except ValueError as exc:
        return None, str(exc)


def plan_skin(
    controller,
    skin: SkinInfo,
    root: str | Path,
    *,
    partial: bool = False,
    single: bool = False,
) -> dict:
    """匹配一个皮肤的素材并返回覆盖报告；不写入任何文件。"""
    skin_dir = find_skin_dir(root, skin, single=single)
    if skin_dir is None:
        raise ValueError(
            f"{skin.controller}: 在 {root} 下找不到皮肤目录（可用名称："
            + "、".join(sorted(_skin_folder_names(skin))) + "）"
        )
    index = index_assets(skin_dir)
    actions = []
    for action in skin.actions:
        asset, error = _match(index, [action], folders=True)
        if asset is None:
            asset, error = _match(index, [action], folders=False)
        actions.append({
            "name": action, "kind": "action", "asset": str(asset) if asset else "",
            "status": "matched" if asset else ("ambiguous" if error else "missing"),
            "error": error, "required": True,
        })
    portraits = []
    for target in portrait_targets(controller, skin):
        asset, error = _match(index, _portrait_aliases(target), folders=False)
        portraits.append({
            "name": target.name, "kind": "portrait", "asset": str(asset) if asset else "",
            "status": "matched" if asset else ("ambiguous" if error else "missing"),
            "error": error, "required": target.required,
            "bundles": list(target.bundles),
        })
    rows = actions + portraits
    missing = [row["name"] for row in rows if row["required"] and not row["asset"]]
    ambiguous = [row for row in rows if row["status"] == "ambiguous"]
    available = [row for row in rows if row["asset"]]
    return {
        "hero_id": skin.hero_id,
        "skin": skin.key,
        "label": skin.label,
        "controller": skin.controller,
        "bundle": skin.bundle,
        "assets_root": str(skin_dir),
        "actions": actions,
        "portraits": portraits,
        "missing": missing,
        "ambiguous": ambiguous,
        "matched": len(available),
        "can_apply": bool(available) and not ambiguous and (partial or not missing),
        "partial": partial,
    }


def plan_hero(
    controller,
    hero_id: str,
    skins: list[str],
    root: str | Path,
    *,
    partial: bool = False,
    refresh: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    available = {skin.key: skin for skin in list_skins(controller, hero_id, refresh=refresh, progress=progress)}
    if not available:
        raise ValueError(f"没有找到角色 {hero_id} 的皮肤。")
    unknown = [key for key in skins if key not in available]
    if unknown:
        raise ValueError(f"角色 {hero_id} 没有这些皮肤：{', '.join(unknown)}")
    single = len(skins) == 1
    return [
        plan_skin(controller, available[key], root, partial=partial, single=single)
        for key in skins
    ]


def _replace_textures_batch(src_bundle: str | Path, replacements: dict[str, str | Path],
                            out_bundle: str | Path) -> dict[str, tuple[int, int]]:
    env = _load_bundle(src_bundle)
    expected: dict[str, tuple[int, int]] = {}
    for name, image_path in replacements.items():
        data = _texture_of(env, name).read()
        width, height = int(data.m_Width), int(data.m_Height)
        with Image.open(image_path) as opened:
            if int(getattr(opened, "n_frames", 1)) != 1:
                raise ValueError(f"立绘必须是静态图片：{image_path}")
            image = opened.convert("RGBA")
            if image.size != (width, height):
                image = image.resize((width, height), Image.LANCZOS)
        try:
            _set_image(data, image)
        finally:
            image.close()
        expected[name] = (width, height)

    def validate(saved) -> None:
        for name, (width, height) in expected.items():
            data = _texture_of(saved, name).read()
            if (int(data.m_Width), int(data.m_Height)) != (width, height):
                raise ValueError(f"写回校验失败：{name} 尺寸发生变化")

    _atomic_save(env, out_bundle, validate)
    return expected


def _base_bundle(controller, bundle: str) -> Path:
    draft = controller._draft_dir() / bundle
    if draft.is_file():
        return draft
    backup = _data_dir() / "backups" / bundle
    if backup.is_file():
        return backup
    path = controller.bundle_path(bundle)
    if path is None:
        raise ValueError(f"找不到资源包：{bundle}")
    return Path(path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def apply_plans(controller, plans: list[dict], *, note_prefix: str = "整角色替换") -> dict:
    """把已匹配的多个皮肤一次性构建并写入作品集；失败不改动现有作品集。"""
    if not plans:
        raise ValueError("没有选择要替换的皮肤。")
    for plan in plans:
        if not plan.get("can_apply"):
            detail = "、".join(plan.get("missing") or []) or "存在歧义素材"
            raise ValueError(f"{plan['controller']}: 素材不完整（{detail}）")
        if not plan.get("matched"):
            raise ValueError(f"{plan['controller']}: 没有匹配到任何素材。")

    actions_by_bundle: dict[str, dict[str, str]] = {}
    portraits_by_bundle: dict[str, dict[str, str]] = {}
    for plan in plans:
        action_rows = [row for row in plan["actions"] if row["asset"]]
        portrait_rows = [row for row in plan["portraits"] if row["asset"]]
        for row in action_rows:
            actions_by_bundle.setdefault(plan["bundle"], {})[row["name"]] = row["asset"]
        for row in portrait_rows:
            for bundle in row["bundles"]:
                portraits_by_bundle.setdefault(bundle, {})[row["name"]] = row["asset"]

    stage = controller._draft_dir().parent / f"_hero_stage_{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    staged: dict[str, Path] = {}
    items: list[dict] = []
    try:
        for bundle, actions in actions_by_bundle.items():
            target = stage / bundle
            base = staged.get(bundle) or _base_bundle(controller, bundle)
            result = replace_animation_frames_batch(
                base, actions, target, packer="lzma", inline_stream=True
            )
            staged[bundle] = target
            skin_plan = next(plan for plan in plans if plan["bundle"] == bundle)
            for action, texture in result["textures"].items():
                items.append({
                    "kind": "texture", "bundle": bundle, "name": texture,
                    "from_anim": action, "frame_atlas": True,
                    "frames": result.get("frame_counts", {}).get(action, 0),
                    "note": f"{note_prefix} · {skin_plan['hero_id']}/{skin_plan['skin']} · {action}",
                    "hero_batch": f"{skin_plan['hero_id']}/{skin_plan['skin']}",
                    "at": _now(),
                })
        for bundle, portraits in portraits_by_bundle.items():
            target = stage / bundle
            base = staged.get(bundle) or _base_bundle(controller, bundle)
            _replace_textures_batch(base, portraits, target)
            staged[bundle] = target
            for name in portraits:
                skin_plan = next(plan for plan in plans
                                 if any(row["name"] == name for row in plan["portraits"]))
                items.append({
                    "kind": "texture", "bundle": bundle, "name": name,
                    "note": f"{note_prefix} · {skin_plan['hero_id']}/{skin_plan['skin']} · {name}",
                    "hero_batch": f"{skin_plan['hero_id']}/{skin_plan['skin']}",
                    "at": _now(),
                })

        for item in items:
            controller._check_sequence_overlap(item["bundle"], [item["name"]], "texture", item["name"])
        check_visual = getattr(controller, "_check_visual_bundle_overlap", None)
        if callable(check_visual):
            for bundle in staged:
                check_visual(bundle)

        draft_dir = controller._draft_dir()
        rollback = stage / "_rollback"
        rollback.mkdir()
        backups: dict[str, Path] = {}
        created: set[str] = set()
        original_items = list(controller.draft_items)
        try:
            for bundle, source in staged.items():
                destination = draft_dir / bundle
                if destination.is_file():
                    backup = rollback / bundle
                    shutil.copy2(destination, backup)
                    backups[bundle] = backup
                else:
                    created.add(bundle)
                os.replace(source, destination)
            keys = {(item["bundle"], item["name"], item["kind"]) for item in items}
            kept = [item for item in controller.draft_items
                    if (item.get("bundle"), item.get("name"), item.get("kind")) not in keys]
            controller.draft_items = kept + items
            controller._save_draft()
        except Exception:
            for bundle, backup in backups.items():
                os.replace(backup, draft_dir / bundle)
            for bundle in created:
                (draft_dir / bundle).unlink(missing_ok=True)
            controller.draft_items = original_items
            try:
                controller._save_draft()
            except OSError:
                pass
            raise
        controller.log(f"整角色替换已加入作品集：{len(plans)} 个皮肤、{len(items)} 项")
        return {
            "items": items,
            "bundles": sorted(staged),
            "skins": [f"{plan['hero_id']}/{plan['skin']}" for plan in plans],
            "draft_items": len(controller.draft_items),
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_template(root: str | Path, skins: list[SkinInfo], controller=None) -> dict:
    """生成素材目录模板和每个皮肤的清单，不写任何游戏文件。"""
    root = Path(root)
    created = []
    for skin in skins:
        folder = root / skin.hero_id / skin.key
        (folder / "portraits").mkdir(parents=True, exist_ok=True)
        (folder / "actions").mkdir(parents=True, exist_ok=True)
        lines = [
            f"{skin.hero_id} / {skin.key}（{skin.label}）素材清单",
            "",
            "动作：放 <动作名>.gif，或 <动作名>/0001.png 这样的 PNG 帧目录。",
        ]
        lines += [f"  {action}.gif   或   {action}/0001.png …" for action in skin.actions]
        if controller is not None:
            lines += ["", "立绘：放下面的文件名（或简写）到 portraits/。"]
            for target in portrait_targets(controller, skin):
                mark = "必需" if target.required else "可选"
                aliases = " / ".join(_portrait_aliases(target)[:2])
                lines.append(f"  {target.name}.png   （{mark}；简写 {aliases}）")
        (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        created.append(str(folder))
    return {"root": str(root), "skins": created}


def main(argv: list[str] | None = None) -> int:
    """命令行入口：列皮肤、生成模板、检查素材或写入作品集。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m astral_party_auto.modkit.hero_skins",
        description="整角色 / 全皮肤批量替换：立绘 + 局内小人全部动作。",
    )
    parser.add_argument("action", choices=("list", "template", "plan", "apply"))
    parser.add_argument("--hero", required=True, help="角色 ID，例如 101")
    parser.add_argument("--skins", default="", help="皮肤列表，例如 base,01,02,Max；默认全部")
    parser.add_argument("--assets", type=Path, help="素材根目录；template 时为输出目录")
    parser.add_argument("--partial", action="store_true", help="只替换已提供的动作/立绘")
    parser.add_argument("--refresh", action="store_true", help="强制重新扫描皮肤")
    args = parser.parse_args(argv)

    from ..mod_controller import ModController

    controller = ModController(lambda message: None)
    controller.refresh_detection()
    if not controller.has_game:
        print("没有检测到游戏目录。")
        return 2
    skins = list_skins(controller, args.hero, refresh=args.refresh)
    if not skins:
        print(f"没有找到角色 {args.hero} 的皮肤。")
        return 2
    selected = [key.strip() for key in args.skins.split(",") if key.strip()] or [skin.key for skin in skins]
    by_key = {skin.key: skin for skin in skins}
    unknown = [key for key in selected if key not in by_key]
    if unknown:
        print(f"角色 {args.hero} 没有这些皮肤：{', '.join(unknown)}")
        return 2

    if args.action == "list":
        print(json.dumps([{
            "skin": skin.key, "label": skin.label, "controller": skin.controller,
            "bundle": skin.bundle, "actions": list(skin.actions),
        } for skin in skins], ensure_ascii=False, indent=2))
        return 0
    if args.assets is None:
        print("这个操作需要 --assets。")
        return 2
    if args.action == "template":
        print(json.dumps(write_template(args.assets, [by_key[key] for key in selected], controller),
                         ensure_ascii=False, indent=2))
        return 0
    plans = plan_hero(controller, args.hero, selected, args.assets, partial=args.partial)
    if args.action == "plan":
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0 if all(plan["can_apply"] for plan in plans) else 1
    print(json.dumps(apply_plans(controller, plans), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
