"""整角色批量替换回归：皮肤发现、素材匹配、模板生成与原子写入。

默认只使用临时目录和模拟资源，不读写真实游戏文件。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from astral_party_auto.mod_controller import ModController
from astral_party_auto.modkit import hero_skins

HERO = "101"
CORE = (
    "UT_Hero_Bust_101_0", "UT_Hero_Bust_101_1", "UT_Hero_Card2_101",
    "UT_Hero_Card_101", "UT_Hero_LevelUp_101", "UT_Hero_ProfilePhoto_101",
    "UT_Hero_RolePhoto_101", "UT_Hero_ThinCard_101",
)


class HeroSkinChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jixing-hero-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.draft = self.root / "draft"
        self.draft.mkdir()
        self.saved: list[list[dict]] = []
        self.bundles = {
            "ctrl.bundle": self.root / "ctrl.bundle",
            "ctrl01.bundle": self.root / "ctrl01.bundle",
            "bust0.bundle": self.root / "bust0.bundle",
            "bust1.bundle": self.root / "bust1.bundle",
            "other.bundle": self.root / "other.bundle",
        }
        for path in self.bundles.values():
            path.write_bytes(b"base")
        self.controller = object.__new__(ModController)
        self.controller.typed_index = {
            "anim": {
                "ctrl.bundle": ["Idle", "Walk-Back"],
                "ctrl01.bundle": ["Idle"],
            },
            "texture": {
                "bust0.bundle": ["UT_Hero_Bust_101_0", "UT_Hero_Bust_101_01_0"],
                "bust1.bundle": ["UT_Hero_Bust_101_1"],
                "other.bundle": [*CORE[2:], "UT_Hero_Card_101_01_sfw"],
            },
        }
        self.controller._index_source_key = "selfcheck"
        self.controller.aa_dirs = (self.root / "cache", self.root / "base")
        self.controller.bundle_path = lambda name: self.bundles.get(name)
        self.controller._draft_dir = lambda: self.draft
        self.controller.draft_items = []
        self.controller._save_draft = lambda: self.saved.append(list(self.controller.draft_items))
        self.controller._check_sequence_overlap = lambda *args: None
        self.controller.log = lambda _message: None
        self.cache_patch = patch.object(hero_skins, "_data_dir", return_value=self.root)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        (self.root / "hero_controllers.json").write_text(json.dumps({
            "source_key": "selfcheck",
            "controllers": {
                "Hero101_Controller": [{"bundle": "ctrl.bundle", "path": str(self.bundles["ctrl.bundle"]), "source": "cache"}],
                "Hero101_01_Controller": [{"bundle": "ctrl01.bundle", "path": str(self.bundles["ctrl01.bundle"]), "source": "base"}],
            },
        }), encoding="utf-8")

    def skin(self, key="base"):
        return next(skin for skin in hero_skins.list_skins(self.controller, HERO) if skin.key == key)

    def assets(self, *, skip: str | None = None):
        folder = self.root / "assets" / HERO / "base"
        (folder / "portraits").mkdir(parents=True, exist_ok=True)
        (folder / "actions").mkdir(parents=True, exist_ok=True)
        for action in ("Idle", "Walk-Back"):
            if action == skip:
                continue
            frames = [Image.new("RGBA", (32, 32), (index * 80, 40, 200, 255)) for index in range(2)]
            frames[0].save(folder / "actions" / f"{action}.gif", save_all=True,
                           append_images=frames[1:], duration=80, loop=0)
        for name in CORE:
            if name == skip:
                continue
            Image.new("RGBA", (32, 32), (200, 30, 40, 255)).save(folder / "portraits" / f"{name}.png")
        return folder

    def test_parse_portrait_covers_skins_aliases_and_variants(self):
        self.assertEqual(hero_skins.parse_portrait("UT_Hero_Bust_101_0", HERO),
                         ("base", "Bust", 0, None, False))
        self.assertEqual(hero_skins.parse_portrait("UT_Hero_Bust_101_01_1", HERO),
                         ("01", "Bust", 1, None, False))
        self.assertEqual(hero_skins.parse_portrait("UT_Hero_Card_101_01_sfw", HERO),
                         ("01", "Card", None, None, True))
        self.assertEqual(hero_skins.parse_portrait("UT_Hero_ProfilePhoto_1019", HERO),
                         ("base", "ProfilePhoto", None, 9, False))
        self.assertIsNone(hero_skins.parse_portrait("UT_Hero_ProfilePhoto_1020", HERO))
        self.assertIsNone(hero_skins.parse_portrait("UT_Hero_Bust_102_0", HERO))
        self.assertIsNone(hero_skins.parse_portrait("UT_Card_101", HERO))

    def test_list_skins_uses_cache_and_labels(self):
        skins = hero_skins.list_skins(self.controller, HERO)
        self.assertEqual([skin.key for skin in skins], ["base", "01"])
        self.assertEqual(skins[0].label, "基础")
        self.assertEqual(skins[1].label, "皮肤 01")
        self.assertEqual(skins[0].actions, ("Idle", "Walk-Back"))
        self.assertEqual(skins[1].bundle, "ctrl01.bundle")

    def test_find_skin_dir_prefers_skin_folder_over_hero_folder(self):
        root = self.root / "assets2" / HERO
        (root / "base").mkdir(parents=True)
        skin = self.skin()
        self.assertEqual(hero_skins.find_skin_dir(root.parent, skin, single=True), root / "base")

    def test_plan_reports_missing_and_partial(self):
        folder = self.assets(skip="Walk-Back")
        plan = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent, single=True)
        self.assertFalse(plan["can_apply"])
        self.assertIn("Walk-Back", plan["missing"])
        partial = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent,
                                       single=True, partial=True)
        self.assertTrue(partial["can_apply"])
        self.assertEqual(partial["matched"], 9)  # Idle + 8 立绘

    def test_plan_rejects_ambiguous_asset_names(self):
        folder = self.assets()
        (folder / "actions" / "extra").mkdir()
        Image.new("RGBA", (16, 16), "red").save(folder / "actions" / "extra" / "Idle.gif")
        plan = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent, single=True)
        self.assertFalse(plan["can_apply"])
        self.assertTrue(plan["ambiguous"])

    def test_template_creates_skin_folders_and_readme(self):
        result = hero_skins.write_template(self.root / "template", [self.skin()], self.controller)
        folder = Path(result["skins"][0])
        self.assertTrue((folder / "portraits").is_dir())
        self.assertTrue((folder / "actions").is_dir())
        text = (folder / "README.txt").read_text(encoding="utf-8")
        self.assertIn("Idle.gif", text)
        self.assertIn("UT_Hero_Bust_101_0", text)

    def test_apply_writes_only_after_success(self):
        folder = self.assets()
        plan = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent, single=True)
        self.assertTrue(plan["can_apply"])

        def fake_batch(base, actions, target, **kwargs):
            Path(target).write_bytes(b"built")
            return {"textures": {action: f"texture-{action}" for action in actions},
                    "frame_counts": {action: 2 for action in actions}, "frames": 2 * len(actions)}

        def fake_portraits(base, replacements, target):
            Path(target).write_bytes(b"portraits")
            return {name: (32, 32) for name in replacements}

        with patch.object(hero_skins, "replace_animation_frames_batch", side_effect=fake_batch), \
                patch.object(hero_skins, "_replace_textures_batch", side_effect=fake_portraits):
            result = hero_skins.apply_plans(self.controller, [plan])
        self.assertEqual(len(result["items"]), 2 + 8)
        self.assertEqual(len(self.controller.draft_items), 10)
        self.assertTrue((self.draft / "ctrl.bundle").is_file())
        self.assertTrue((self.draft / "bust0.bundle").is_file())
        self.assertEqual(self.controller.draft_items[0]["frame_atlas"], True)
        self.assertTrue(self.saved)

    def test_apply_failure_leaves_draft_untouched(self):
        folder = self.assets()
        plan = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent, single=True)
        self.controller.draft_items = [{"kind": "texture", "bundle": "ctrl.bundle", "name": "old"}]
        with patch.object(hero_skins, "replace_animation_frames_batch", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                hero_skins.apply_plans(self.controller, [plan])
        self.assertEqual(self.controller.draft_items, [{"kind": "texture", "bundle": "ctrl.bundle", "name": "old"}])
        self.assertEqual(list(self.draft.iterdir()), [])

    def test_incomplete_plan_is_rejected_before_building(self):
        folder = self.assets(skip="Idle")
        plan = hero_skins.plan_skin(self.controller, self.skin(), folder.parent.parent, single=True)
        with self.assertRaisesRegex(ValueError, "素材不完整"):
            hero_skins.apply_plans(self.controller, [plan])
        self.assertEqual(self.controller.draft_items, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
