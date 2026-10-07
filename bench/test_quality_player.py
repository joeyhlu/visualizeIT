"""Synthetic accessibility/content regression for the comparison-page help."""
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import DEFAULT, patch

from . import quality_player as player


class _HelpDisclosureParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_help = False
        self.in_summary = False
        self.details_attrs = None
        self.summary_attrs = None
        self.summary_text = []
        self.help_text = []
        self.help_tags = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "details" and attributes.get("id") == "objectSelectionHelp":
            self.in_help = True
            self.details_attrs = attributes
            return
        if not self.in_help:
            return
        self.help_tags.append(tag)
        if tag == "summary" and self.summary_attrs is None:
            self.in_summary = True
            self.summary_attrs = attributes

    def handle_endtag(self, tag):
        if not self.in_help:
            return
        if tag == "summary":
            self.in_summary = False
        elif tag == "details":
            self.in_help = False

    def handle_data(self, data):
        if self.in_help:
            self.help_text.append(data)
            if self.in_summary:
                self.summary_text.append(data)


class ObjectSelectionHelpTests(unittest.TestCase):
    def _build_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "artifacts" / "video-60"
            output_root = root / "artifacts" / "model-quality"
            for alias in ("keyboard", "mug", "ranch"):
                source_case = source_root / alias
                source_case.mkdir(parents=True)
                (source_case / "player.json").write_text(
                    json.dumps({"frames": []}), encoding="utf-8")
                annotation_case = output_root / "annotations" / alias
                annotation_case.mkdir(parents=True)
                (annotation_case / "annotations.json").write_text(
                    json.dumps({"frames": []}), encoding="utf-8")

            artifacts = {}
            with patch.multiple(
                    player,
                    ROOT=root,
                    CACHE=root / ".cache",
                    _load_mug_pnp_comparison=DEFAULT,
                    _load_bottle_pose_summary=DEFAULT,
                    _bottle_pose_summary_html=DEFAULT,
                    _load_bottle_patch_calibration_summary=DEFAULT,
                    _bottle_patch_calibration_summary_html=DEFAULT,
            ) as patched:
                patched["_load_mug_pnp_comparison"].return_value = None
                patched["_load_bottle_pose_summary"].return_value = None
                patched["_bottle_pose_summary_html"].return_value = ""
                patched["_load_bottle_patch_calibration_summary"].return_value = None
                patched["_bottle_patch_calibration_summary_html"].return_value = ""
                with patch.object(
                        player, "write_artifact",
                        side_effect=lambda path, text: artifacts.__setitem__(Path(path), text)):
                    player.build()
            return artifacts[output_root / "index.html"]

    def test_generated_page_has_collapsed_accessible_explanation_and_scope(self):
        page = self._build_page()
        self.assertIn(player._OFFLINE_REPLAY_NOTICE, page)
        help_start = page.index('<details id="objectSelectionHelp"')
        cases_start = page.index('<div class="controls" id="cases"></div>')
        self.assertLess(page.index(player._OFFLINE_REPLAY_NOTICE), help_start)
        self.assertLess(help_start, cases_start)

        parser = _HelpDisclosureParser()
        parser.feed(page)
        self.assertIsNotNone(parser.details_attrs)
        self.assertNotIn("open", parser.details_attrs,
                         "the help should start collapsed")
        self.assertEqual("How object selection works", "".join(parser.summary_text).strip())
        self.assertIn("min-height:44px", parser.summary_attrs.get("style", ""))

        text = " ".join(" ".join(parser.help_text).lower().split())
        required_explanations = (
            "saved comparisons for the keyboard, white mug, and dressing bottle",
            "fixed textured 3d model",
            "do not identify or scan a new object",
            "six prompts once",
            "three points on the object and three on the hand or background",
            "sam 2 propagates the mask forward",
            "automatic cnos recovery",
            "no scored correction clicks are used",
            "controlled mode gets one onboarding pose",
            "complete mode estimates its initialization with foundpose and gotrack",
            "mask pixels mark image regions",
            "do not confirm that the 3d model has the correct pose",
            "replays precomputed results",
            "scanning an arbitrary object on a phone is not implemented yet",
        )
        for explanation in required_explanations:
            with self.subTest(explanation=explanation):
                self.assertIn(explanation, text)

        forbidden_controls_or_assets = {
            "a", "button", "canvas", "iframe", "img", "input", "select", "video"
        }
        self.assertTrue(forbidden_controls_or_assets.isdisjoint(parser.help_tags))


if __name__ == "__main__":
    unittest.main()
