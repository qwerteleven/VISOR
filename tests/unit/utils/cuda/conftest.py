import sys
from pathlib import Path

FAKES_DIR = Path(__file__)
sys.path.insert(0, str(FAKES_DIR))

for name in list(sys.modules):
    if name == "tensorrt" or name.startswith("cuda"):
        del sys.modules[name]

import tests.unit.utils.cuda.tensorrt as _fake_trt  # noqa: E402
import tests.unit.utils.cuda.cuda as _fake_cuda  # noqa: E402
import tests.unit.utils.cuda.cudart as _fake_cudart  # noqa: E402

sys.modules["tensorrt"] = _fake_trt
sys.modules["cuda.cuda"] = _fake_cuda
sys.modules["cuda.cudart"] = _fake_cudart
