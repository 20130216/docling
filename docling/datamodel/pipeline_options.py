import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
)
from typing_extensions import deprecated

# Import the following for backwards compatibility
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_OLLAMA as granite_vision_vlm_ollama_conversion_options,
    GRANITE_VISION_TRANSFORMERS as granite_vision_vlm_conversion_options,
    SMOLDOCLING_MLX as smoldocling_vlm_mlx_conversion_options,
    SMOLDOCLING_TRANSFORMERS as smoldocling_vlm_conversion_options,
    VlmModelType,
)

_log = logging.getLogger(__name__)


class BaseOptions(BaseModel):
    """Base class for options."""

    kind: ClassVar[str]


class TableFormerMode(str, Enum):
    """Modes for the TableFormer model."""

    FAST = "fast"
    ACCURATE = "accurate"


class TableStructureOptions(BaseModel):
    """Options for the table structure."""

    do_cell_matching: bool = (
        True
        # True:  Matches predictions back to PDF cells. Can break table output if PDF cells
        #        are merged across table columns.
        # False: Let table structure model define the text cells, ignore PDF cells.
        # 总结：提高表格提取精度，但需注意合并单元格的影响
        # 适用场景: 适合 PDF 单元格划分清晰、表格结构规整的文档，能够提高表格提取的精确度。
        # 潜在风险: 英文注释指出，如果 PDF 中的单元格跨列合并（merged across table columns），可能会导致表格输出错误（如内容错位或丢失）。
        # 影响: 开启此功能可增强表格提取的准确性，但对复杂或不规范的表格（例如跨列合并的单元格）可能引发问题，需要文档布局支持。
    )
    mode: TableFormerMode = TableFormerMode.ACCURATE


class OcrOptions(BaseOptions):
    """OCR options."""

    lang: List[str]
    force_full_page_ocr: bool = False  # If enabled a full page OCR is always applied
    bitmap_area_threshold: float = (
        0.05  # percentage of the area for a bitmap to processed with OCR
    )


class RapidOcrOptions(OcrOptions):
    """Options for the RapidOCR engine."""

    kind: ClassVar[Literal["rapidocr"]] = "rapidocr"

    # English and chinese are the most commly used models and have been tested with RapidOCR.
    lang: List[str] = [
        "english",
        "chinese",
    ]
    # However, language as a parameter is not supported by rapidocr yet
    # and hence changing this options doesn't affect anything.

    # For more details on supported languages by RapidOCR visit
    # https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/

    # For more details on the following options visit
    # https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/

    text_score: float = 0.5  # same default as rapidocr

    use_det: Optional[bool] = None  # same default as rapidocr
    use_cls: Optional[bool] = None  # same default as rapidocr
    use_rec: Optional[bool] = None  # same default as rapidocr

    print_verbose: bool = False  # same default as rapidocr

    det_model_path: Optional[str] = None  # same default as rapidocr
    cls_model_path: Optional[str] = None  # same default as rapidocr
    rec_model_path: Optional[str] = None  # same default as rapidocr
    rec_keys_path: Optional[str] = None  # same default as rapidocr

    model_config = ConfigDict(
        extra="forbid",
    )


class EasyOcrOptions(OcrOptions):
    """Options for the EasyOCR engine."""

    kind: ClassVar[Literal["easyocr"]] = "easyocr"
    lang: List[str] = ["fr", "de", "es", "en"]

    use_gpu: Optional[bool] = None

    confidence_threshold: float = 0.5

    model_storage_directory: Optional[str] = None
    recog_network: Optional[str] = "standard"
    download_enabled: bool = True

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class TesseractCliOcrOptions(OcrOptions):
    """Options for the TesseractCli engine."""

    kind: ClassVar[Literal["tesseract"]] = "tesseract"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    tesseract_cmd: str = "tesseract"
    path: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )


class TesseractOcrOptions(OcrOptions):
    """Options for the Tesseract engine."""

    kind: ClassVar[Literal["tesserocr"]] = "tesserocr"
    lang: List[str] = ["fra", "deu", "spa", "eng"]
    path: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )


class OcrMacOptions(OcrOptions):
    """Options for the Mac OCR engine."""

    kind: ClassVar[Literal["ocrmac"]] = "ocrmac"
    lang: List[str] = ["fr-FR", "de-DE", "es-ES", "en-US"]
    recognition: str = "accurate"
    framework: str = "vision"

    model_config = ConfigDict(
        extra="forbid",
    )


class PictureDescriptionBaseOptions(BaseOptions):
    batch_size: int = 8
    scale: float = 2

    picture_area_threshold: float = (
        0.05  # percentage of the area for a picture to processed with the models
    )


class PictureDescriptionApiOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["api"]] = "api"

    url: AnyUrl = AnyUrl("http://localhost:8000/v1/chat/completions")
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    timeout: float = 20
    concurrency: int = 1

    prompt: str = "Describe this image in a few sentences."
    provenance: str = ""


class PictureDescriptionVlmOptions(PictureDescriptionBaseOptions):
    kind: ClassVar[Literal["vlm"]] = "vlm"

    repo_id: str
    prompt: str = "Describe this image in a few sentences."
    # Config from here https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    generation_config: Dict[str, Any] = dict(max_new_tokens=200, do_sample=False)

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


# SmolVLM
smolvlm_picture_description = PictureDescriptionVlmOptions(
    repo_id="HuggingFaceTB/SmolVLM-256M-Instruct"
)

# GraniteVision
granite_picture_description = PictureDescriptionVlmOptions(
    repo_id="ibm-granite/granite-vision-3.1-2b-preview",
    prompt="What is shown in this image?",
)


# Define an enum for the backend options
class PdfBackend(str, Enum):
    """Enum of valid PDF backends."""

    PYPDFIUM2 = "pypdfium2"
    DLPARSE_V1 = "dlparse_v1"
    DLPARSE_V2 = "dlparse_v2"
    DLPARSE_V4 = "dlparse_v4"


# Define an enum for the ocr engines
@deprecated("Use ocr_factory.registered_enum")
class OcrEngine(str, Enum):
    """Enum of valid OCR engines."""

    EASYOCR = "easyocr"
    TESSERACT_CLI = "tesseract_cli"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"
    RAPIDOCR = "rapidocr"


class PipelineOptions(BaseModel):
    """Base pipeline options."""

    create_legacy_output: bool = (
        True  # This default will be set to False on a future version of docling
    )
    document_timeout: Optional[float] = None
    accelerator_options: AcceleratorOptions = AcceleratorOptions()
    enable_remote_services: bool = False
    allow_external_plugins: bool = False


class PaginatedPipelineOptions(PipelineOptions):
    artifacts_path: Optional[Union[Path, str]] = None

    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False

# 专注于 VLM 模型（本地或云端），处理复杂文档（如学术论文、技术报告），包含嵌套表格、图像、复杂布局；
# 需要多模态解析，结合图像和文本理解（如 OCR 表格、描述图像内容）。
# 场景: 学术论文、复杂报告，使用 GPT-4.1 或 SmolDocling。
class VlmPipelineOptions(PaginatedPipelineOptions):
    generate_page_images: bool = True  # 强制将 PDF 转换为图像，因为 VLM 模型依赖图像输入。
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models)
    )
    # force_backend_text=True: 使用后端提取的文本（如 OCR），而非 VLM 生成的文本。
    # If True, text from backend will be used instead of generated text
    
    # vlm_options: 支持本地（InlineVlmOptions）或远程（ApiVlmOptions）模型。
    vlm_options: Union[InlineVlmOptions, ApiVlmOptions] = (
        smoldocling_vlm_conversion_options
    )

# 场景：扫描件、简单 PDF 等简单或结构化程度较高的 PDF，需提取文本、表格等；
# 功能: 提供丰富的功能开关（如表格提取、OCR、代码/公式增强、图片分类/描述）；
# PdfPipelineOptions：功能复杂，支持多种结构化提取（表格、代码、公式等），配置灵活，适合本地处理。
class PdfPipelineOptions(PaginatedPipelineOptions):
    """Options for the PDF pipeline."""

    do_table_structure: bool = True  # True: perform table structure extraction  #控制特定功能（如表格提取、OCR、代码/公式增强）
                                     # 启用表格结构提取，适合财务或数据表。
    do_ocr: bool = True  # True: perform OCR, replace programmatic PDF text
    do_code_enrichment: bool = False  # True: perform code OCR ;
                                      # PDF 中的代码片段（例如编程代码）进行专门的 OCR 处理，以优化代码内容的提取和格式化。
    do_formula_enrichment: bool = False  # True: perform formula OCR, return Latex code
    do_picture_classification: bool = False  # True: classify pictures in documents
    do_picture_description: bool = False  # True: run describe pictures in documents
    force_backend_text: bool = (
        False  # (To be used with vlms, or other generative models) #默认使用生成文本
    )
    # If True, text from backend will be used instead of generated text

    table_structure_options: TableStructureOptions = TableStructureOptions() #配置表格
    ocr_options: OcrOptions = EasyOcrOptions()                               #和 OCR 处理
    picture_description_options: PictureDescriptionBaseOptions = (
        smolvlm_picture_description
    )

    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False
    generate_table_images: bool = Field(
        default=False,
        deprecated=(
            "Field `generate_table_images` is deprecated. "
            "To obtain table images, set `PdfPipelineOptions.generate_page_images = True` "
            "before conversion and then use the `TableItem.get_image` function."
        ),
    )

    generate_parsed_pages: bool = False

# 功能: 定义 PDF 管道类型（standard 或 vlm）
class PdfPipeline(str, Enum):
    STANDARD = "standard"   # 传统 PDF 解析（如 OCR、表格提取）
    VLM = "vlm"   # 使用 VLM（如 GPT-4.1）解析
