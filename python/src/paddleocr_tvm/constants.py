from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "python" / ".artifacts"
DEFAULT_DICT_PATH = (
    REPO_ROOT / "reference" / "PaddleOCR" / "ppocr" / "utils" / "dict" / "ppocrv5_dict.txt"
)

DET_LIMIT_SIDE_LEN = 960
DET_LIMIT_TYPE = "max"
DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.6
DET_DB_UNCLIP_RATIO = 1.5
DET_IMAGE_SHAPE = (3, 960, 960)
REC_IMAGE_SHAPE = (3, 48, 320)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a bundled upstream model."""

    key: str
    filename: str
    url: str
    input_name: str = "x"


MODEL_SPECS: dict[str, ModelSpec] = {
    "mobile_det": ModelSpec(
        key="mobile_det",
        filename="PP-OCRv5_mobile_det_infer.tar",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
        ),
    ),
    "mobile_rec": ModelSpec(
        key="mobile_rec",
        filename="PP-OCRv5_mobile_rec_infer.tar",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar"
        ),
    ),
}
