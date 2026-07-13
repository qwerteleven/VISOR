import sys
from pathlib import Path

FAKES_DIR = Path(__file__)
sys.path.insert(0, str(FAKES_DIR))

for name in list(sys.modules):
    if name in ("demo_trt_webcam", "utils", "utils.io", "sam_api"):
        del sys.modules[name]

import tests.unit.object_segmentation.sam_model as _fake_sam_model  # noqa: E402
import tests.unit.object_segmentation.io as _fake_utils_io  # noqa: E402

sys.modules["demo_trt_webcam"] = _fake_sam_model
sys.modules["utils.io"] = _fake_utils_io
