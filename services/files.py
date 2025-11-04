from typing import List, Dict, BinaryIO
from abc import ABC, abstractmethod
from pathlib import Path
import base64
import os
from PIL import Image
import io
from pydub import AudioSegment
from PyPDF2 import PdfReader
import pandas as pd
import requests
from loguru import logger
import zipfile
import xml.etree.ElementTree as ET


class BaseFile(ABC):
    def __init__(self, name: str=""):
        super().__init__()
        self.name = name

    @property
    def extension(self) -> str:
        _, file_extension = os.path.splitext(self.name)
        file_extension = file_extension.lower().replace(".", "")
        if file_extension == "jpg":
            file_extension = "jpeg"
        return file_extension

    @property
    def bytes_io(self) -> BinaryIO:
        output = io.BytesIO(self.file_bytes)
        output.name = self.name
        return output


class DocumentFile(BaseFile):
    def __init__(self, name: str=""):
        super().__init__(name=name)

    @property
    def size(self) -> int:
        if hasattr(self, 'bytes'):
            return len(self.bytes)
        elif hasattr(self, 'text'):
            return len(self.text)
        return 0
    
    @property
    def base64(self) -> str:
        if hasattr(self, 'bytes'):
            return base64.b64encode(self.bytes).decode('utf-8')
        logger.warning("Base64 encoding not available for this document type.")
        return ""

class TextDocumentFile(DocumentFile):
    def __init__(self, text: str, name: str=""):
        super().__init__(name=name)
        self.text = text

    @classmethod
    def from_string(cls, text: str, name: str="") -> 'TextDocumentFile':
        return cls(text, name)


class PDFDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str=""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str="") -> 'PDFDocumentFile':
        return cls(bytes, name)

    @property
    def text(self) -> str:
        try:
            pdf_stream = io.BytesIO(self.bytes)
            reader = PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return ""
    
    @property
    def number_of_pages(self) -> int:
        try:
            pdf_stream = io.BytesIO(self.bytes)
            reader = PdfReader(pdf_stream)
            return len(reader.pages)
        except Exception as e:
                return 0
    
class ExcelDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str=""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str="") -> 'ExcelDocumentFile':
        return cls(bytes, name)

    @property
    def text(self) -> str:
        # Read Excel file from bytes using pandas
        excel_file = io.BytesIO(self.bytes)
        # Get all sheet names
        excel = pd.ExcelFile(excel_file)
        sheet_names = excel.sheet_names
        
        # Read each sheet and combine their content
        text_parts = []
        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            text_parts.append(f"Sheet: {sheet}\n{df.to_string(index=False)}\n")
        
        text = "\n".join(text_parts)
        return text

class WordDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str=""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str="") -> 'WordDocumentFile':
        return cls(bytes, name)
    
    @property
    def text(self) -> str:
        try:
            doc_stream = io.BytesIO(self.bytes)
            with zipfile.ZipFile(doc_stream) as document:
                xml_bytes = document.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs: List[str] = []
            for paragraph in root.iter(f"{namespace}p"):
                texts = [node.text for node in paragraph.iter(f"{namespace}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
        except Exception:
            logger.exception("Failed to extract text from Word document.")
            return ""

class PowerPointDocumentFile(DocumentFile):
    def __init__(self, bytes: BinaryIO, name: str=""):
        super().__init__(name=name)
        self.bytes = bytes

    @classmethod
    def from_bytes(cls, bytes: BinaryIO, name: str="") -> 'PowerPointDocumentFile':
        return cls(bytes, name)

    @property
    def text(self) -> str:
        try:
            ppt_stream = io.BytesIO(self.bytes)
            with zipfile.ZipFile(ppt_stream) as presentation:
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
                slides_text: List[str] = []
                for _, slide_name in slide_entries:
                    xml_bytes = presentation.read(slide_name)
                    root = ET.fromstring(xml_bytes)
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
    def from_url(cls, url: str) -> 'MediaFile':
        file_name = os.path.basename(url)
        with open(url, "rb") as media_file:
            return cls(media_file.read(), file_name)
        
    @classmethod
    def from_web_url(cls, url: str) -> 'MediaFile':
        response = requests.get(url)
        if response.status_code == 200:
            file_name = os.path.basename(url)
            # Try to convert the image to PNG format
            try:
                # Create BytesIO object from the response content
                image_buffer = io.BytesIO(response.content)
                # Try to open as an image
                pil_image = Image.open(image_buffer)
                # Convert to PNG
                png_buffer = io.BytesIO()
                pil_image.save(png_buffer, format="PNG")
                png_buffer.seek(0)
                # Update file name with .png extension
                file_name = os.path.splitext(file_name)[0] + ".png"
                # Return the PNG bytes
                return cls(png_buffer.getvalue(), file_name)
            except Exception:
                raise ValueError(f"Failed to convert file to PNG.")
        else:
            raise ValueError(f"Failed to fetch file from URL: {url}")

    @classmethod
    def from_bytes(cls, file_bytes: bytes, file_name: str) -> 'MediaFile':
        return cls(file_bytes, file_name)
    
    @classmethod
    def from_base64(cls, base64_str: str, file_name: str) -> 'MediaFile':
        file_bytes = base64.b64decode(base64_str)
        return cls(file_bytes, file_name)

    @property
    def base64(self) -> str:
        return base64.b64encode(self.file_bytes).decode('utf-8')

class ImageFile(MediaFile):
    
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name)

    @classmethod
    def from_pil_image(cls, pil_image: Image.Image, file_name: str = "image.png") -> 'ImageFile':
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        file_bytes = buffer.getvalue()
        return cls(file_bytes=file_bytes, name=file_name)
    
    @property
    def pil_image(self):
        # Open the image using PIL
        pil_image = Image.open(self.bytes_io)
        return pil_image

    @property
    def size(self) -> int:
        return len(self.file_bytes)

class AudioFile(MediaFile):
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name)
        
        if self.extension.lower() != "mp3":
            self.file_bytes = self.convert_to_mp3()

    def convert_to_mp3(self) -> bytes:
        # Use BytesIO to handle the file in memory
        audio_stream = io.BytesIO(self.file_bytes)

        try:
            audio = AudioSegment.from_file(audio_stream, format=self.extension)
        except Exception as e:
            raise ValueError(f"Error loading audio stream to AudioSegment: {e}")
        
        wav_stream = io.BytesIO()
        audio.export(wav_stream, format="mp3")
        wav_stream.seek(0)
        return wav_stream.read()

class VideoFile(MediaFile):
    
    def __init__(self, file_bytes: bytes, name: str):
        super().__init__(file_bytes, name=name)
