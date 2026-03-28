import base64
import io
import os
import xml.etree.ElementTree as ET
import zipfile
from abc import ABC
from pathlib import Path
from typing import BinaryIO, Literal

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


class BaseFile(ABC):
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    @property
    def extension(self) -> str:
        return _normalize_extension(self.name)

    @property
    def bytes_io(self) -> BinaryIO:
        output = io.BytesIO(self.file_bytes)
        output.name = self.name
        return output


class DocumentFile(BaseFile):
    def __init__(self, name: str = ""):
        super().__init__(name=name)

    @property
    def size(self) -> int:
        if hasattr(self, "bytes"):
            return len(self.bytes)
        if hasattr(self, "text"):
            return len(self.text)
        return 0

    @property
    def base64(self) -> str:
        if hasattr(self, "bytes"):
            return base64.b64encode(self.bytes).decode("utf-8")
        logger.warning("Base64 encoding not available for this document type.")
        return ""


class TextDocumentFile(DocumentFile):
    def __init__(self, text: str, name: str = ""):
        super().__init__(name=name)
        self.text = text

    @classmethod
    def from_file(cls, file_path: Path) -> "TextDocumentFile":
        with open(file_path, "r") as file:
            return cls(file.read(), name=file_path.name)

    @classmethod
    def from_string(cls, text: str, name: str = "") -> "TextDocumentFile":
        return cls(text, name)


class PDFDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str = ""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str = "") -> "PDFDocumentFile":
        return cls(bytes, name)

    @classmethod
    def from_file(cls, name: str) -> "PDFDocumentFile":
        return cls(_read_binary_file(name), name=name)

    @property
    def text(self) -> str:
        try:
            reader = PdfReader(io.BytesIO(self.bytes))
            return "".join(f"{page.extract_text()}\n" for page in reader.pages)
        except Exception:
            return ""

    @property
    def number_of_pages(self) -> int:
        try:
            reader = PdfReader(io.BytesIO(self.bytes))
            return len(reader.pages)
        except Exception:
            return 0


class ExcelDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str = ""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str = "") -> "ExcelDocumentFile":
        return cls(bytes, name)

    @classmethod
    def from_file(cls, name: str) -> "ExcelDocumentFile":
        return cls(_read_binary_file(name), name=name)

    @property
    def text(self) -> str:
        excel = pd.ExcelFile(io.BytesIO(self.bytes))
        text_parts = []
        for sheet_name in excel.sheet_names:
            dataframe = excel.parse(sheet_name=sheet_name)
            text_parts.append(f"Sheet: {sheet_name}\n{dataframe.to_string(index=False)}\n")
        return "\n".join(text_parts)


class WordDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str = ""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str = "") -> "WordDocumentFile":
        return cls(bytes, name)

    @classmethod
    def from_file(cls, name: str) -> "WordDocumentFile":
        return cls(_read_binary_file(name), name=name)

    @property
    def text(self) -> str:
        try:
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            root = _read_zip_xml(self.bytes, "word/document.xml")
            paragraphs = []
            for paragraph in root.iter(f"{namespace}p"):
                texts = [node.text for node in paragraph.iter(f"{namespace}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
        except Exception:
            logger.exception("Failed to extract text from Word document.")
            return ""


class PowerPointDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str = ""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str = "") -> "PowerPointDocumentFile":
        return cls(bytes, name)

    @classmethod
    def from_file(cls, name: str) -> "PowerPointDocumentFile":
        return cls(_read_binary_file(name), name=name)

    @property
    def text(self) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(self.bytes)) as presentation:
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

                return "\n\n".join(slides_text)
        except Exception:
            logger.exception("Failed to extract text from PowerPoint document.")
            return ""


class MediaFile(BaseFile):
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(name=name)
        self.file_bytes = file_bytes

    @classmethod
    def from_url(cls, url: str) -> "MediaFile":
        file_name = os.path.basename(url)
        with open(url, "rb") as media_file:
            return cls(media_file.read(), file_name)

    @classmethod
    def from_web_url(cls, url: str) -> "MediaFile":
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file from URL: {url}")

        file_name = os.path.basename(url)

        try:
            image_buffer = io.BytesIO(response.content)
            pil_image = Image.open(image_buffer)
            png_buffer = io.BytesIO()
            pil_image.save(png_buffer, format="PNG")
            png_buffer.seek(0)
            file_name = os.path.splitext(file_name)[0] + ".png"
            return cls(png_buffer.getvalue(), file_name)
        except Exception as error:
            raise ValueError("Failed to convert file to PNG.") from error

    @classmethod
    def from_bytes(cls, file_bytes: bytes, file_name: str) -> "MediaFile":
        return cls(file_bytes, file_name)

    @classmethod
    def from_base64(cls, base64_str: str, file_name: str) -> "MediaFile":
        return cls(base64.b64decode(base64_str), file_name)

    @property
    def base64(self) -> str:
        return base64.b64encode(self.file_bytes).decode("utf-8")


class ImageFile(MediaFile):
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name)

    @classmethod
    def from_pil_image(
        cls,
        pil_image: Image.Image,
        file_name: str = "image.png",
    ) -> "ImageFile":
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return cls(file_bytes=buffer.getvalue(), name=file_name)

    @property
    def pil_image(self):
        return Image.open(self.bytes_io)

    @property
    def size(self) -> int:
        return len(self.file_bytes)


class AudioFile(MediaFile):
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name)

        if self.extension.lower() != "mp3":
            self.file_bytes = self.convert_to_mp3()

    def convert_to_mp3(self) -> bytes:
        audio_stream = io.BytesIO(self.file_bytes)

        try:
            audio = AudioSegment.from_file(audio_stream, format=self.extension)
        except Exception as error:
            raise ValueError(f"Error loading audio stream to AudioSegment: {error}")

        mp3_stream = io.BytesIO()
        audio.export(mp3_stream, format="mp3")
        mp3_stream.seek(0)
        return mp3_stream.read()


class VideoFile(MediaFile):
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name=name)
