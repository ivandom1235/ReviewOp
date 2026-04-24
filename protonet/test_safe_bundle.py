from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from protonet.code.export_bundle import export_safe_model_bundle
from protonet.code.runtime_infer import load_safe_bundle_payload


class SafeBundleTests(unittest.TestCase):
    def temp_dir(self):
        root = Path.cwd() / "protonet" / "output" / "test_safe_bundle_tmp"
        path = root / self._testMethodName
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_export_safe_bundle_writes_json_and_state_files(self) -> None:
        tmp = self.temp_dir()
        if True:
            bundle_dir = export_safe_model_bundle(
                output_dir=tmp,
                config={"encoder_backend": "bow", "bow_dim": 2},
                encoder={"backend": "bow", "hidden_size": 2},
                projection_state_dict={"layers.0.weight": torch.ones(2, 2), "layers.4.weight": torch.ones(2, 2)},
                prototype_bank={"labels": ["battery__negative"], "prototypes": torch.ones(1, 2)},
                temperature=1.0,
                novelty_calibration={"version": "test"},
            )

            self.assertTrue((bundle_dir / "config.json").exists())
            self.assertTrue((bundle_dir / "encoder.json").exists())
            self.assertTrue((bundle_dir / "label_map.json").exists())
            self.assertTrue((bundle_dir / "projection_state.pt").exists())
            self.assertTrue((bundle_dir / "prototype_bank.pt").exists())
            self.assertEqual(json.loads((bundle_dir / "label_map.json").read_text())["labels"], ["battery__negative"])

    def test_load_safe_bundle_payload_uses_directory_contract(self) -> None:
        tmp = self.temp_dir()
        if True:
            bundle_dir = export_safe_model_bundle(
                output_dir=tmp,
                config={"encoder_backend": "bow", "bow_dim": 2},
                encoder={"backend": "bow", "hidden_size": 2},
                projection_state_dict={"layers.0.weight": torch.ones(2, 2), "layers.4.weight": torch.ones(2, 2)},
                prototype_bank={"labels": ["battery__negative"], "prototypes": torch.ones(1, 2)},
                temperature=1.0,
                novelty_calibration={},
            )

            payload = load_safe_bundle_payload(bundle_dir)

            self.assertEqual(payload["bundle_version"], "2.0")
            self.assertEqual(payload["config"]["encoder_backend"], "bow")
            self.assertEqual(payload["prototype_bank"]["labels"], ["battery__negative"])
            self.assertTrue(torch.equal(payload["prototype_bank"]["prototypes"], torch.ones(1, 2)))


if __name__ == "__main__":
    unittest.main()
