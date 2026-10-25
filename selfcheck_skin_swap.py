"""同角色两个皮肤互换的回归：计划、公共项、原子构建与安装调用。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from astral_party_auto.mod_controller import ModController
from astral_party_auto.modkit import hero_skins, skin_swap

HERO = "101"


class SkinSwapChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jixing-swap-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.made = self.root / "made"
        self.made.mkdir()
        self.bundles = {
            "ctrl_a.bundle": self.root / "ctrl_a.bundle",
            "ctrl_b.bundle": self.root / "ctrl_b.bundle",
            "bust_a.bundle": self.root / "bust_a.bundle",
            "bust_b.bundle": self.root / "bust_b.bundle",
        }
        for path in self.bundles.values():
            path.write_bytes(b"base")
        (self.root / "hero_controllers.json").write_text(json.dumps({
            "source_key": "selfcheck",
            "controllers": {
                "Hero101_Controller": [{"bundle": "ctrl_a.bundle", "path": str(self.bundles["ctrl_a.bundle"]), "source": "base"}],
                "Hero101_01_Controller": [{"bundle": "ctrl_b.bundle", "path": str(self.bundles["ctrl_b.bundle"]), "source": "base"}],
            },
        }), encoding="utf-8")
        self.controller = object.__new__(ModController)
        self.controller.typed_index = {
            "anim": {
                "ctrl_a.bundle": ["Idle", "Walk-Back"],
                "ctrl_b.bundle": ["Idle", "Walk-Back", "Extra"],
            },
            "texture": {
                "bust_a.bundle": ["UT_Hero_Bust_101_0", "UT_Hero_Bust_101_1"],
                "bust_b.bundle": ["UT_Hero_Bust_101_01_0", "UT_Hero_Bust_101_01_1"],
            },
        }
        self.controller._index_source_key = "selfcheck"
        self.controller.aa_dirs = (self.root / "cache", self.root / "base")
        self.controller.bundle_path = lambda name: self.bundles.get(name)
        self.controller.original_bundle_path = lambda name: self.bundles.get(name)
        self.controller.install = Mock()
        self.controller.log = lambda _message: None
        self.data_patch = patch.object(hero_skins, "_data_dir", return_value=self.root)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.made_patch = patch.object(skin_swap, "_made_dir", return_value=self.made)
        self.made_patch.start()
        self.addCleanup(self.made_patch.stop)

    def test_plan_common_actions_and_portraits(self):
        plan = skin_swap.plan_swap(self.controller, HERO, "base", "01")
        self.assertEqual(plan["actions"], ["Idle", "Walk-Back"])
        self.assertEqual(plan["only_b"], ["Extra"])
        self.assertEqual(len(plan["portraits"]), 2)
        self.assertTrue(plan["can_apply"])

    def test_same_skin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "两个不同的皮肤"):
            skin_swap.plan_swap(self.controller, HERO, "base", "base")

    def test_swap_builds_pack_and_calls_install(self):
        def fake_render(bundle, action, out_dir):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for index in range(2):
                Image.new("RGBA", (8, 8), (index * 100, 40, 90, 255)).save(Path(out_dir) / f"{index:05d}.png")
            return 2

        def fake_batch(base, replacements, target, **kwargs):
            Path(target).write_bytes(b"swapped")
            return {"frame_counts": {name: 2 for name in replacements}}

        def fake_extract(bundle, name, out_png):
            Image.new("RGBA", (8, 8), (20, 30, 40, 255)).save(out_png)
            return (8, 8)

        def fake_portraits(base, replacements, target):
            Path(target).write_bytes(b"portrait")
            return {name: (8, 8) for name in replacements}

        with patch.object(skin_swap, "_render_frames", side_effect=fake_render), \
                patch.object(skin_swap, "replace_animation_frames_batch", side_effect=fake_batch), \
                patch.object(skin_swap, "_extract_portrait", side_effect=fake_extract), \
                patch.object(skin_swap, "_replace_textures_batch", side_effect=fake_portraits):
            result = skin_swap.swap_skins(self.controller, HERO, "base", "01", install=True)
        self.assertTrue(result["installed"])
        self.assertEqual(result["actions"], 2)
        self.assertEqual(result["portraits"], 4)
        pack = Path(result["pack"])
        self.assertTrue((pack / "ctrl_a.bundle").is_file())
        self.assertTrue((pack / "ctrl_b.bundle").is_file())
        self.assertTrue((pack / "bust_a.bundle").is_file())
        self.assertTrue((pack / "bust_b.bundle").is_file())
        self.assertTrue((pack / "README.txt").is_file())
        self.controller.install.assert_called_once()

    def test_swap_failure_leaves_no_pack(self):
        with patch.object(skin_swap, "_render_frames", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                skin_swap.swap_skins(self.controller, HERO, "base", "01")
        self.assertEqual(list(self.made.glob("皮肤互换_*")), [])
        self.controller.install.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
