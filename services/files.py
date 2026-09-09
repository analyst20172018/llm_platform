import base64
import io
import mimetypes
import os
import warnings
import xml.etree.ElementTree as ET
import zipfile
from abc import ABC
from pathlib import Path
from typing import BinaryIO, Literal
from urllib.parse import unquote, urlsplit

import pandas as pd
import requests
from loguru import logger
from PIL import Image
from pydub import AudioSegment
from PyPDF2 import PdfReader

FileType = Literal[
    "text", "pdf", "excel", "word", "powerpoint",
    "image", "audio", "video", "unknown",
]

_IMAGE_EXTS = {
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif"
}
_AUDIO_EXTS = {
    "mp3", "wav", "m4a", "aac", "ogg", "flac", "opus", "wma", "aiff", "aif"
}
_VIDEO_EXTS = {
    "mp4", "mov", "mkv", "avi", "webm", "m4v", "mpg", "mpeg", "3gp"
}
_TEXT_EXTS = {
    "txt", "md", "rtf", "csv", "tsv", "json", "yaml", "yml", "xml", "log", "html", "htm"
}
_EXCEL_EXTS = {"xls", "xlsx", "xlsm", "xlsb"}
_WORD_EXTS = {"docx"}
_PPT_EXTS = {"pptx"}


def _normalize_extension(file_name: str) -> str:
    if not file_name:
        return ""

    clean_name = file_name.split("?", 1)[0].split("#", 1)[0]
    extension = Path(clean_name).suffix.lower().lstrip(".")
    return "jpeg" if extension == "jpg" else extension


def _read_binary_file(file_name: str) -> bytes:
    with open(file_name, "rb") as file:
        return file.read()


def _read_zip_xml(document_bytes: bytes, file_name: str) -> ET.Element:
    with zipfile.ZipFile(io.BytesIO(document_bytes)) as archive:
        return ET.fromstring(archive.read(file_name))


def define_file_type(file_name: str) -> FileType:
    """
    Determine a broad file type category from the filename extension.
    Returns: 'image' | 'audio' | 'video' | 'pdf' | 'excel' | 'word' | 'powerpoint' | 'text' | 'unknown'
    """
    if not file_name:
        return "unknown"

    extension = _normalize_extension(file_name)

    if extension in _IMAGE_EXTS:
        return "image"
    if extension in _AUDIO_EXTS:
        return "audio"
    if extension in _VIDEO_EXTS:
        return "video"
    if extension == "pdf":
        return "pdf"
    if extension in _EXCEL_EXTS:
        return "excel"
    if extension in _WORD_EXTS:
        return "word"
    if extension in _PPT_EXTS:
        return "powerpoint"
    if extension in _TEXT_EXTS:
        return "text"

    return "unknown"


class DocumentExtractionError(ValueError):
    """Document text could not be recovered reliably."""


class DocumentExtractionWarning(UserWarning):
    """A binary document is being reduced to text, losing layout and visuals."""


def _warn_text_extraction(file):
    warnings.warn(
        f"Extracting text from '{file.name}'; layout, images and other non-text content are omitted.",
        DocumentExtractionWarning,
        stacklevel=3,
    )


class BaseFile(ABC):
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    @property
    def extension(self) -> str:
        return _normalize_extension(self.name)

    @property
    def mime_type(self) -> str:
        return getattr(self, "_mime_type", None) or mimetypes.guess_type(self.name)[0] or "application/octet-stream"

    @property
    def bytes_io(self) -> BinaryIO:
        output = io.BytesIO(self.data)
        output.name = self.name
        return output


class DocumentFile(BaseFile):
    def __init__(self, name: str = ""):
        super().__init__(name=name)

    @property
    def size(self) -> int:
        if hasattr(self, "data"):
            return len(self.data)
        if hasattr(self, "text"):
            return len(self.text)
        return 0

    @property
    def base64(self) -> str:
        if hasattr(self, "data"):
            return base64.b64encode(self.data).decode("utf-8")
        logger.warning("Base64 encoding not available for this document type.")
        return ""


class TextDocumentFile(DocumentFile):
    def __init__(self, text: str, name: str = ""):
        super().__init__(name=name)
        self.text = text

    @property
    def data(self) -> bytes:
        return self.text.encode("utf-8")

    @classmethod
    def from_file(cls, file_path: Path) -> "TextDocumentFile":
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as file:
            return cls(file.read(), name=file_path.name)

    @classmethod
    def from_string(cls, text: str, name: str = "") -> "TextDocumentFile":
        return cls(text, name)


class ByteDocumentFile(DocumentFile):
    """Base for byte-backed documents: stores raw ``data`` and exposes text via a
    subclass ``text`` property. Subclasses only differ in how they extract text."""

    def __init__(self, data: bytes, name: str = ""):
        super().__init__(name=name)
        self.data = data

    @classmethod
    def from_bytes(cls, data: bytes, file_name: str = "") -> "ByteDocumentFile":
        return cls(data, file_name)

    @classmethod
    def from_file(cls, name: str) -> "ByteDocumentFile":
        return cls(_read_binary_file(name), name=name)


class PDFDocumentFile(ByteDocumentFile):
    @property
    def mime_type(self) -> str:
        return "application/pdf"

    @property
    def text(self) -> str:
        try:
            reader = PdfReader(io.BytesIO(self.data))
            pages = [page.extract_text() or "" for page in reader.pages]
            if any(not text.strip() for text in pages):
                raise DocumentExtractionError(
                    f"PDF '{self.name}' contains pages without extractable text; use native PDF input or OCR."
                )
            _warn_text_extraction(self)
            return "\n".join(pages)
        except DocumentExtractionError:
            raise
        except Exception as error:
            raise DocumentExtractionError(f"Failed to extract PDF text: {self.name}") from error

    @property
    def number_of_pages(self) -> int:
        try:
            reader = PdfReader(io.BytesIO(self.data))
            return len(reader.pages)
        except Exception as error:
            raise DocumentExtractionError(f"Failed to read PDF page count: {self.name}") from error


class ExcelDocumentFile(ByteDocumentFile):
    @property
    def text(self) -> str:
        _warn_text_extraction(self)
        excel = pd.ExcelFile(io.BytesIO(self.data))
        text_parts = []
        for sheet_name in excel.sheet_names:
            dataframe = excel.parse(sheet_name=sheet_name)
            text_parts.append(f"Sheet: {sheet_name}\n{dataframe.to_string(index=False)}\n")
        return "\n".join(text_parts)


class WordDocumentFile(ByteDocumentFile):
    @property
    def text(self) -> str:
        try:
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            root = _read_zip_xml(self.data, "word/document.xml")
            paragraphs = []
            for paragraph in root.iter(f"{namespace}p"):
                texts = [node.text for node in paragraph.iter(f"{namespace}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            _warn_text_extraction(self)
            return "\n".join(paragraphs)
        except Exception as error:
            raise DocumentExtractionError(f"Failed to extract Word text: {self.name}") from error


class PowerPointDocumentFile(ByteDocumentFile):
    @property
    def text(self) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(self.data)) as presentation:
                presentation.getinfo("ppt/presentation.xml")
                slide_entries = []
                for name in presentation.namelist():
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                        try:
                            index = int(name.rsplit("slide", 1)[1].split(".xml")[0])
                        except (IndexError, ValueError):
                            index = 0
                        slide_entries.append((index, name))

                slide_entries.sort(key=lambda entry: entry[0])
                namespace = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
                slides_text = []

                for _, slide_name in slide_entries:
                    root = ET.fromstring(presentation.read(slide_name))
                    texts = [node.text for node in root.iter(f"{namespace}t") if node.text]
                    if texts:
                        slides_text.append(" ".join(texts))

                _warn_text_extraction(self)
                return "\n\n".join(slides_text)
        except Exception as error:
            raise DocumentExtractionError(f"Failed to extract PowerPoint text: {self.name}") from error


class MediaFile(BaseFile):
    def __init__(self, data: bytes, name: str):
        super().__init__(name=name)
        self.data = data

    @classmethod
    def from_path(cls, url: str) -> "MediaFile":
        file_name = os.path.basename(url)
        with open(url, "rb") as media_file:
            return cls(media_file.read(), file_name)

    @classmethod
    def from_web_url(cls, url: str) -> "MediaFile":
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file from URL: {url}")

        file_name = unquote(Path(urlsplit(url).path).name) or "download"
        return cls(response.content, file_name)

    @classmethod
    def from_bytes(cls, data: bytes, file_name: str) -> "MediaFile":
        return cls(data, file_name)

    @classmethod
    def from_base64(cls, base64_str: str, file_name: str) -> "MediaFile":
        return cls(base64.b64decode(base64_str), file_name)

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("utf-8")


class ImageFile(MediaFile):
    def __init__(self, data: bytes, name: str):
        super().__init__(data, name)

    @classmethod
    def from_pil_image(
        cls,
        pil_image: Image.Image,
        file_name: str = "image.png",
    ) -> "ImageFile":
        buffer = io.BytesIO()
        extension = _normalize_extension(file_name) or "png"
        formats = {
            "png": "PNG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF",
            "bmp": "BMP", "tiff": "TIFF", "tif": "TIFF",
        }
        if extension not in formats:
            raise ValueError(f"Unsupported image output extension: {extension}")
        if not Path(file_name).suffix:
            file_name = (file_name or "image") + ".png"
        if extension == "jpeg":
            pil_image = pil_image.convert("RGB")
        pil_image.save(buffer, format=formats[extension])
        buffer.seek(0)
        return cls(data=buffer.getvalue(), name=file_name)

    @property
    def mime_type(self) -> str:
        try:
            with Image.open(self.bytes_io) as image:
                return Image.MIME.get(image.format) or super().mime_type
        except (OSError, ValueError):
            return super().mime_type

    @property
    def pil_image(self):
        return Image.open(self.bytes_io)

    @property
    def size(self) -> int:
        return len(self.data)


class AudioFile(MediaFile):
    def __init__(self, data: bytes, name: str):
        super().__init__(data, name)

    def as_mp3(self) -> "AudioFile":
        """Return MP3 input for providers that require it, without changing this file."""
        if self.mime_type in ("audio/mpeg", "audio/mp3"):
            return self
        return AudioFile(self.convert_to_mp3(), str(Path(self.name).with_suffix(".mp3")))

    def convert_to_mp3(self) -> bytes:
        audio_stream = io.BytesIO(self.data)

        try:
            audio = AudioSegment.from_file(audio_stream, format=self.extension)
        except Exception as error:
            raise ValueError(f"Error loading audio stream to AudioSegment: {error}")

        mp3_stream = io.BytesIO()
        audio.export(mp3_stream, format="mp3")
        mp3_stream.seek(0)
        return mp3_stream.read()


class VideoFile(MediaFile):
    def __init__(self, data: bytes, name: str):
        super().__init__(data, name=name)


class BinaryFile(MediaFile):
    """An arbitrary artifact whose original bytes must be preserved."""


class FailedFile(BaseFile):
    """A remote artifact that could not be downloaded; reference supports retry."""

    def __init__(self, name: str, reference: dict, error: str):
        super().__init__(name)
        self.reference = dict(reference)
        self.error = error


def file_from_bytes(data: bytes, filename: str) -> BaseFile:
    """Classify an artifact without extracting text or transforming its bytes."""
    if not isinstance(data, bytes):
        raise TypeError("File content must be bytes")
    classes = {
        "image": ImageFile, "audio": AudioFile, "video": VideoFile,
        "pdf": PDFDocumentFile, "excel": ExcelDocumentFile,
        "word": WordDocumentFile, "powerpoint": PowerPointDocumentFile,
    }
    kind = define_file_type(filename)
    if kind == "text":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return BinaryFile(data, filename)
        return TextDocumentFile(text, filename)
    return classes.get(kind, BinaryFile)(data, filename)
