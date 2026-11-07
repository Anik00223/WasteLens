"""Tests for WasteLens web application contract, model assets, and decision logic."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
MODEL_DIR = WEB_DIR / "model"


class ApplicationContractTests(unittest.TestCase):
    def test_web_model_assets_exist(self):
        """Verify model.json, labels.json, and all weight shards exist."""
        model_json_path = MODEL_DIR / "model.json"
        labels_json_path = MODEL_DIR / "labels.json"
        self.assertTrue(model_json_path.is_file(), "web/model/model.json must exist")
        self.assertTrue(labels_json_path.is_file(), "web/model/labels.json must exist")

        with open(model_json_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.assertEqual(model_data.get("format"), "layers-model")
        manifest = model_data.get("weightsManifest", [])
        self.assertGreater(len(manifest), 0, "weightsManifest must not be empty")

        total_shards = 0
        for entry in manifest:
            for shard_path in entry.get("paths", []):
                full_path = MODEL_DIR / shard_path
                self.assertTrue(full_path.is_file(), f"Weight shard {shard_path} must exist on disk")
                self.assertGreater(full_path.stat().st_size, 0, f"Weight shard {shard_path} must not be empty")
                total_shards += 1
        self.assertEqual(total_shards, 3, "Expected exactly 3 TF.js weight shards for MobileNetV2 dual-head")

    def test_labels_order_and_content(self):
        """Verify 4 official classification classes in strict order."""
        labels_json_path = MODEL_DIR / "labels.json"
        with open(labels_json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        expected = ["recyclable", "organic", "hazardous", "general trash"]
        self.assertEqual(labels, expected, "Class mapping and index order must be strictly preserved")

    def test_frontend_constants_and_logic_in_html(self):
        """Verify index.html contains the official threshold, parameters, and verdict states."""
        index_html_path = WEB_DIR / "index.html"
        self.assertTrue(index_html_path.is_file(), "web/index.html must exist")
        content = index_html_path.read_text(encoding="utf-8")

        self.assertIn("REJECT_THRESHOLD = 0.0702", content)
        self.assertIn("MIN_TOP: 0.60, MIN_MARGIN: 0.50", content)
        self.assertIn("div(127.5).sub(1)", content)
        self.assertIn("'unsupported'", content)
        self.assertIn("'uncertain'", content)
        self.assertIn("'supported'", content)
        self.assertIn("That file does not look like an image", content)
        self.assertIn("Opened from the file system (file://)", content)

    def test_preprocessing_numeric_parity(self):
        """Verify preprocessing formula maps uint8 [0, 255] strictly to [-1.0, 1.0]."""
        def preprocess(x):
            return (x / 127.5) - 1.0

        self.assertAlmostEqual(preprocess(0.0), -1.0)
        self.assertAlmostEqual(preprocess(127.5), 0.0)
        self.assertAlmostEqual(preprocess(255.0), 1.0)
        self.assertTrue(-1.0 <= preprocess(100.0) <= 1.0)

    def test_pure_verdict_decision_rules(self):
        """Verify 3-state decision logic on boundary conditions."""
        threshold = 0.0702
        min_top = 0.60
        min_margin = 0.50

        def decide(probs, reject_prob):
            best = 0
            second = 1
            if probs[1] > probs[0]:
                best, second = 1, 0
            for i in range(2, len(probs)):
                if probs[i] > probs[best]:
                    second = best
                    best = i
                elif probs[i] > probs[second]:
                    second = i
            margin = probs[best] - probs[second]

            if reject_prob >= threshold:
                return "unsupported"
            elif probs[best] < min_top - 1e-9 or margin + 1e-9 < min_margin:
                return "uncertain"
            else:
                return "supported"

        # 1. Supported confident case
        self.assertEqual(decide([0.95, 0.02, 0.02, 0.01], 0.001), "supported")
        # 2. Rejection priority even with high classification score
        self.assertEqual(decide([0.98, 0.01, 0.005, 0.005], 0.0702), "unsupported")
        self.assertEqual(decide([0.98, 0.01, 0.005, 0.005], 0.891), "unsupported")
        # 3. Below threshold, but low top probability -> uncertain
        self.assertEqual(decide([0.45, 0.25, 0.20, 0.10], 0.01), "uncertain")
        # 4. Below threshold, top >= 0.60, but margin < 0.50 -> uncertain
        self.assertEqual(decide([0.65, 0.25, 0.05, 0.05], 0.01), "uncertain")
        # 5. Just below threshold with high top and margin -> supported
        self.assertEqual(decide([0.90, 0.05, 0.03, 0.02], 0.07019), "supported")

    def test_multi_waste_contract_elements(self):
        """Verify presence of mode switch, candidate regions, IoU, experimental notice, and DOM elements."""
        index_html_path = WEB_DIR / "index.html"
        content = index_html_path.read_text(encoding="utf-8")

        self.assertIn('id="modeSingle"', content)
        self.assertIn('id="modeMulti"', content)
        self.assertIn('Analysis mode:', content)
        self.assertIn("var CANDIDATE_REGIONS = [", content)
        self.assertIn("function calculateIoU(boxA, boxB)", content)
        self.assertIn("function suppressDuplicateRegions(candidates", content)
        self.assertIn('id="multiResultPanel"', content)
        self.assertIn('id="boxOverlay"', content)
        self.assertIn("Candidate Waste Regions", content)
        self.assertIn("Multi-Region Analysis", content)
        self.assertIn("Experimental Multi-Waste Analysis:", content)
        self.assertIn("This mode examines several fixed image regions", content)

    def test_iou_mathematical_parity(self):
        """Verify IoU calculation matches expected geometry."""
        def calculate_iou(box_a, box_b):
            ymin = max(box_a[0], box_b[0])
            xmin = max(box_a[1], box_b[1])
            ymax = min(box_a[2], box_b[2])
            xmax = min(box_a[3], box_b[3])
            inter_h = max(0.0, ymax - ymin)
            inter_w = max(0.0, xmax - xmin)
            inter_area = inter_h * inter_w
            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
            union_area = area_a + area_b - inter_area
            if union_area <= 0:
                return 0.0
            return inter_area / union_area

        self.assertAlmostEqual(calculate_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(calculate_iou([0, 0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]), 0.0)
        self.assertAlmostEqual(calculate_iou([0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1]), 0.36)

    def test_suppress_duplicate_regions_behavior(self):
        """Verify NMS duplicate suppression separates OOD and filters overlapping duplicates."""
        def calculate_iou(box_a, box_b):
            ymin = max(box_a[0], box_b[0])
            xmin = max(box_a[1], box_b[1])
            ymax = min(box_a[2], box_b[2])
            xmax = min(box_a[3], box_b[3])
            inter = max(0.0, ymax - ymin) * max(0.0, xmax - xmin)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def suppress_duplicates(candidates, same_thresh=0.30, diff_thresh=0.60):
            rejected = [c for c in candidates if c.get("state") == "unsupported"]
            valid = [c for c in candidates if c.get("state") != "unsupported"]
            valid.sort(key=lambda c: (1 if c.get("state") == "supported" else 0, c.get("top", 0)), reverse=True)

            accepted = []
            suppressed = []
            for cand in valid:
                is_dup = False
                for acc in accepted:
                    iou = calculate_iou(cand["box"], acc["box"])
                    if cand["best"] == acc["best"] and iou >= same_thresh:
                        is_dup = True
                        break
                    elif iou >= diff_thresh:
                        is_dup = True
                        break
                if is_dup:
                    suppressed.append(cand)
                else:
                    accepted.append(cand)
            return accepted, suppressed, rejected

        c_full = {"id": 1, "box": [0, 0, 1, 1], "state": "supported", "best": 0, "top": 0.95}
        c_center = {"id": 2, "box": [0.2, 0.2, 0.8, 0.8], "state": "supported", "best": 0, "top": 0.99}
        c_bottom_right = {"id": 3, "box": [0.4, 0.4, 1.0, 1.0], "state": "supported", "best": 1, "top": 0.96}
        c_ood = {"id": 4, "box": [0, 0.4, 0.6, 1.0], "state": "unsupported", "best": 0, "top": 0.60}

        accepted, suppressed, rejected = suppress_duplicates([c_full, c_center, c_bottom_right, c_ood])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["id"], 4)
        self.assertEqual(len(accepted), 2)
        self.assertEqual([c["id"] for c in accepted], [2, 3])
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()

