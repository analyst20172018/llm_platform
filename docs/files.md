# Files Module Documentation

## Overview

The `files.py` module provides classes for handling different types of files within the LLM Platform. It defines a hierarchy of file types including documents, images, audio, and video files, with specialized functionality for each type. These file classes are used throughout the platform to manage and process various file formats when interacting with language models.

## Class Hierarchy

```
BaseFile
  ├── DocumentFile
  │     ├── TextDocumentFile
  │     ├── PDFDocumentFile
  │     └── ExcelDocumentFile
  └── MediaFile
        ├── ImageFile
        ├── AudioFile
        └── VideoFile
```

## Classes

### BaseFile

`BaseFile` is the abstract base class for all file types in the platform.

#### Constructor

```python
def __init__(self, name: str = "")
```

**Parameters:**
- `name`: The name of the file

#### Properties

```python
@property
def extension(self) -> str
```
Returns the file extension (normalized to lowercase without the dot).

### DocumentFile

`DocumentFile` is a base class for files containing document content.

#### Constructor

```python
def __init__(self, name: str = "")
```

**Parameters:**
- `name`: The name of the document file

### TextDocumentFile

`TextDocumentFile` represents a text document.

#### Constructor

```python
def __init__(self, text: str, name: str = "")
```

**Parameters:**
- `text`: The text content of the document
- `name`: The name of the document file

#### Class Methods

```python
@classmethod
def from_string(cls, text: str, name: str = "") -> 'TextDocumentFile'
```
Creates a TextDocumentFile from a string.

#### Properties

```python
@property
def size(self) -> int
```
Returns the size of the text content in bytes.

### PDFDocumentFile

`PDFDocumentFile` represents a PDF document.

#### Constructor

```python
def __init__(self, bytes: BinaryIO, name: str = "")
```

**Parameters:**
- `bytes`: The binary content of the PDF file
- `name`: The name of the PDF file

#### Class Methods

```python
@classmethod
def from_bytes(cls, bytes: BinaryIO, name: str = "") -> 'PDFDocumentFile'
```
Creates a PDFDocumentFile from binary data.

#### Properties

```python
@property
def size(self) -> int
```
Returns the size of the PDF in bytes.

```python
@property
def text(self) -> str
```
Extracts and returns the text content from the PDF.

```python
@property
def base64(self) -> str
```
Returns the base64-encoded representation of the PDF.

```python
@property
def number_of_pages(self) -> int
```
Returns the number of pages in the PDF.

### ExcelDocumentFile

`ExcelDocumentFile` represents an Excel document.

#### Constructor

```python
def __init__(self, bytes: BinaryIO, name: str = "")
```

**Parameters:**
- `bytes`: The binary content of the Excel file
- `name`: The name of the Excel file

#### Class Methods

```python
@classmethod
def from_bytes(cls, bytes: BinaryIO, name: str = "") -> 'ExcelDocumentFile'
```
Creates an ExcelDocumentFile from binary data.

#### Properties

```python
@property
def size(self) -> int
```
Returns the size of the Excel file in bytes.

```python
@property
def text(self) -> str
```
Extracts and returns the text content from all sheets in the Excel file.

```python
@property
def base64(self) -> str
```
Returns the base64-encoded representation of the Excel file.

### MediaFile

`MediaFile` is a base class for files containing media content.

#### Constructor

```python
def __init__(self, file_bytes: bytes, name: str)
```

**Parameters:**
- `file_bytes`: The binary content of the media file
- `name`: The name of the media file

#### Class Methods

```python
@classmethod
def from_url(cls, url: str) -> 'MediaFile'
```
Creates a MediaFile from a URL.

```python
@classmethod
def from_bytes(cls, file_bytes: bytes, file_name: str) -> 'MediaFile'
```
Creates a MediaFile from binary data.

```python
@classmethod
def from_base64(cls, base64_str: str, file_name: str) -> 'MediaFile'
```
Creates a MediaFile from a base64-encoded string.

#### Properties

```python
@property
def base64(self) -> str
```
Returns the base64-encoded representation of the media file.

### ImageFile

`ImageFile` represents an image file.

#### Constructor

```python
def __init__(self, file_bytes: bytes, name: str)
```

**Parameters:**
- `file_bytes`: The binary content of the image file
- `name`: The name of the image file

#### Properties

```python
@property
def pil_image(self)
```
Returns a PIL Image object created from the image data.

```python
@property
def size(self) -> int
```
Returns the size of the image in bytes.

### AudioFile

`AudioFile` represents an audio file.

#### Constructor

```python
def __init__(self, file_bytes: bytes, name: str)
```

**Parameters:**
- `file_bytes`: The binary content of the audio file
- `name`: The name of the audio file

#### Methods

```python
def convert_to_mp3(self) -> bytes
```
Converts the audio file to MP3 format and returns the binary data.

### VideoFile

`VideoFile` represents a video file (minimal implementation).

#### Constructor

```python
def __init__(self, name: str = "")
```

**Parameters:**
- `name`: The name of the video file

## Usage Examples

### Working with Text Documents

```python
from llm_platform.services.files import TextDocumentFile

# Create a text document
text_doc = TextDocumentFile(
    text="This is a sample text document.\nIt contains multiple lines.",
    name="sample.txt"
)

# Get document properties
print(f"File name: {text_doc.name}")
print(f"File extension: {text_doc.extension}")
print(f"File size: {text_doc.size} bytes")
print(f"Content: {text_doc.text}")
```

### Working with PDF Documents

```python
from llm_platform.services.files import PDFDocumentFile
import io

# Read a PDF file
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

# Create a PDF document
pdf_doc = PDFDocumentFile(
    bytes=pdf_bytes,
    name="document.pdf"
)

# Get document properties
print(f"File name: {pdf_doc.name}")
print(f"Number of pages: {pdf_doc.number_of_pages}")
print(f"File size: {pdf_doc.size} bytes")

# Extract text from the PDF
pdf_text = pdf_doc.text
print(f"PDF content (first 100 chars): {pdf_text[:100]}...")
```

### Working with Images

```python
from llm_platform.services.files import ImageFile
from PIL import Image

# Read an image file
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

# Create an image file
image = ImageFile(
    file_bytes=image_bytes,
    name="image.jpg"
)

# Get image properties
print(f"File name: {image.name}")
print(f"File extension: {image.extension}")
print(f"File size: {image.size} bytes")

# Work with the image using PIL
pil_img = image.pil_image
width, height = pil_img.size
print(f"Image dimensions: {width}x{height}")

# Convert to base64 for API requests
base64_str = image.base64
print(f"Base64 (first 50 chars): {base64_str[:50]}...")
```

### Working with Audio Files

```python
from llm_platform.services.files import AudioFile

# Read an audio file
with open("recording.wav", "rb") as f:
    audio_bytes = f.read()

# Create an audio file (automatically converts to MP3)
audio = AudioFile(
    file_bytes=audio_bytes,
    name="recording.wav"
)

# Get audio properties
print(f"File name: {audio.name}")
print(f"File extension: {audio.extension}")

# Get base64 representation for API requests
base64_str = audio.base64
print(f"Base64 (first 50 chars): {base64_str[:50]}...")
```

## Integration with LLM Adapters

The file classes in this module are designed to work seamlessly with the LLM Platform's adapters. When making requests to language models that support multimodal inputs, you can include files as part of the request:

```python
from llm_platform.core.llm_handler import APIHandler
from llm_platform.services.files import ImageFile

# Initialize the LLM handler
handler = APIHandler()

# Create an image file
with open("image.jpg", "rb") as f:
    image = ImageFile(file_bytes=f.read(), name="image.jpg")

# Make a request to an LLM with the image
response = handler.request(
    model="gpt-4-vision",
    prompt="Describe what you see in this image.",
    files=[image]
)

print(response)
```

Each adapter handles the file data appropriately for its specific LLM provider's API requirements.