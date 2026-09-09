"""Offline regressions for attachment preservation, conversion, and failures."""

import asyncio
import base64
import io
import json
import zipfile
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from PyPDF2 import PdfWriter

from llm_platform.adapters.google_adapter import GoogleAdapter
from llm_platform.adapters.openai_adapter import OpenAIAdapter
from llm_platform.adapters.serializers import (
    function_response_to_anthropic,
    function_response_to_openai,
    function_response_to_openai_chat,
)
from llm_platform.services.conversation import Conversation, FunctionResponse, Message
from llm_platform.services.files import (
    AudioFile, BinaryFile, DocumentExtractionError, DocumentExtractionWarning,
    FailedFile, ImageFile, PDFDocumentFile, PowerPointDocumentFile,
    TextDocumentFile, VideoFile, WordDocumentFile, file_from_bytes,
)


def restored(conversation):
    return Conversation.read_from_json(json.loads(json.dumps(conversation.save_to_json())))


def png():
    return ImageFile.from_pil_image(Image.new("RGB", (2, 2)), "test.png")


@pytest.mark.parametrize("name", ["script.py", "bundle.zip", "drawing.svg", "no_extension"])
def test_unknown_generated_artifacts_survive_persistence(name):
    file = OpenAIAdapter._build_file_from_container_data(b"\x00\xfforiginal", name)
    assert isinstance(file, BinaryFile)
    copy = restored(Conversation([Message("assistant", "", files=[file])])).messages[0].files[0]
    assert isinstance(copy, BinaryFile)
    assert (copy.name, copy.data) == (name, b"\x00\xfforiginal")


def test_non_utf8_text_artifact_is_preserved_without_replacement_characters():
    file = file_from_bytes(b"caf\xe9", "answer.txt")
    assert isinstance(file, BinaryFile)
    assert file.data == b"caf\xe9"


def test_tool_attachments_preserved_without_mutating_source():
    source = {"answer": 42, "files": [
        {"type": "pdf", "name": "report.pdf", "source": {
            "type": "base64", "data": base64.b64encode(b"pdf bytes").decode(),
        }},
        {"type": "unknown", "name": "archive.zip", "source": {
            "type": "base64", "data": base64.b64encode(b"zip bytes").decode(),
        }},
    ]}
    result = FunctionResponse("export", source, call_id="call")
    assert "files" in source
    assert result.response == {"answer": 42}
    copy = restored(Conversation([Message("function", "", function_responses=[result])]))
    files = copy.messages[0].function_responses[0].files
    assert [type(file) for file in files] == [PDFDocumentFile, BinaryFile]
    assert [file.data for file in files] == [b"pdf bytes", b"zip bytes"]


@pytest.mark.parametrize("source", [
    {"type": "url", "data": "https://example.test/image"},
    {"type": "base64", "data": "!!!"},
    {"type": "base64"},
])
def test_invalid_tool_sources_raise_explicitly(source):
    with pytest.raises(ValueError):
        FunctionResponse("export", {"files": [{"name": "test.png", "source": source}]})


def test_openai_tool_result_contains_images_and_native_pdfs():
    result = FunctionResponse("export", {"files": [png(), PDFDocumentFile(b"pdf", "report.pdf")]}, call_id="call")
    output = function_response_to_openai(result)
    assert output["call_id"] == "call"
    assert [item["type"] for item in output["output"]] == ["input_text", "input_image", "input_file"]
    assert output["output"][1]["image_url"].startswith("data:image/png;base64,")
    assert output["output"][2]["file_data"] == "data:application/pdf;base64,cGRm"


def test_google_tool_result_uses_images_and_text():
    result = FunctionResponse("export", {"files": [png(), TextDocumentFile("content", "notes.txt")]}, call_id="call")
    output = GoogleAdapter()._function_result_entry(result)
    assert output["call_id"] == "call"
    assert [item["type"] for item in output["result"]] == ["text", "image", "text"]
    assert output["result"][2]["text"] == "notes.txt\ncontent"
    # Validate the result with the real installed SDK, including subcontent types.
    from google.genai._gaos.types.interactions.functionresultstep import FunctionResultStep
    parsed = FunctionResultStep.model_validate(output)
    assert [item.type for item in parsed.result] == ["text", "image", "text"]


@pytest.mark.parametrize("serialize", [
    function_response_to_openai, function_response_to_openai_chat,
    function_response_to_anthropic, GoogleAdapter()._function_result_entry,
])
def test_unsupported_tool_attachments_are_not_silently_omitted(serialize):
    result = FunctionResponse("export", {"files": [BinaryFile(b"zip", "a.zip")]})
    with pytest.raises(ValueError, match="attachment"):
        serialize(result)
    assert result.files[0].data == b"zip"


@pytest.mark.parametrize("extension,format_name", [("jpg", "JPEG"), ("png", "PNG"), ("webp", "WEBP")])
def test_pil_encoding_matches_filename_and_wire_mime(extension, format_name):
    file = ImageFile.from_pil_image(Image.new("RGBA", (2, 2)), f"photo.{extension}")
    with Image.open(file.bytes_io) as image:
        assert image.format == format_name
    assert OpenAIAdapter()._image_data_url(file).startswith(f"data:{file.mime_type};")


def test_mislabeled_image_uses_actual_mime():
    file = ImageFile(png().data, "incorrect.jpg")
    assert file.mime_type == "image/png"


def test_audio_is_preserved_and_conversion_is_explicit(monkeypatch):
    conversion = MagicMock(return_value=b"mp3")
    monkeypatch.setattr(AudioFile, "convert_to_mp3", conversion)
    file = AudioFile(b"wav", "sound.wav")
    conversion.assert_not_called()
    copy = restored(Conversation([Message("assistant", "", files=[file])])).messages[0].files[0]
    assert (copy.name, copy.data) == ("sound.wav", b"wav")
    output = OpenAIAdapter()._convert_file_to_content(file)
    assert output["input_audio"] == {"format": "mp3", "data": "bXAz"}
    assert (file.name, file.data) == ("sound.wav", b"wav")
    assert GoogleAdapter()._file_mime_type(file) != "audio/mp3"


def test_legacy_audio_uses_mp3_mime_without_changing_bytes_or_filename():
    audio = Conversation._deserialize_file({
        "type": "AudioFile", "name": "original.wav", "base64": "bXAz",
    })
    assert (audio.name, audio.data, audio.mime_type) == ("original.wav", b"mp3", "audio/mpeg")
    assert audio.as_mp3() is audio


def test_pdf_mime_does_not_depend_on_filename():
    assert GoogleAdapter()._file_mime_type(PDFDocumentFile(b"pdf")) == "application/pdf"


@pytest.mark.parametrize("cls,name", [(AudioFile, "sound.wav"), (VideoFile, "video.mp4"), (ImageFile, "photo.gif")])
def test_url_loading_preserves_payload_and_removes_query(monkeypatch, cls, name):
    get = MagicMock(return_value=NS(status_code=200, content=b"original bytes"))
    monkeypatch.setattr("llm_platform.services.files.requests.get", get)
    file = cls.from_web_url(f"https://example.test/{name}?signature=abc#fragment")
    assert (file.name, file.data) == (name, b"original bytes")
    assert get.call_args.kwargs["timeout"] == 30


def citation(file_id, filename):
    return {"container_id": "container", "file_id": file_id, "filename": filename}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_download_failures_preserve_answer_deduplicate_and_retry(asynchronous, with_tools):
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    conversation = Conversation([Message("user", "export")])
    annotations = [NS(type="container_file_citation", **item) for item in [
        citation("good", "script.py"), citation("bad", "archive.zip"), citation("good", "script.py"),
    ]]
    response = NS(id="resp", model="test", usage=None, status="completed", output=[
        NS(type="message", role="assistant", content=[NS(type="output_text", text="Done", annotations=annotations)]),
    ])

    def download(**kwargs):
        # The completed assistant message is already recorded at download time.
        assert conversation.messages[-1].content == "Done"
        if kwargs["file_id"] == "bad":
            raise OSError("temporary download failure")
        return NS(content=b"script bytes")

    kwargs = {"functions": [lambda: None]} if with_tools else {}
    if asynchronous:
        adapter._async_client.responses.create = AsyncMock(return_value=response)
        retrieve = AsyncMock(side_effect=download)
        adapter._async_client.containers.files.content.retrieve = retrieve
        message = asyncio.run(adapter.request_llm_async("test", conversation, **kwargs))
    else:
        adapter._client.responses.create.return_value = response
        retrieve = MagicMock(side_effect=download)
        adapter._client.containers.files.content.retrieve = retrieve
        message = adapter.request_llm("test", conversation, **kwargs)
    assert message is conversation.messages[-1]
    assert message.content == "Done"
    assert retrieve.call_count == 2
    assert len(message.files) == 2
    assert isinstance(message.files[0], BinaryFile)
    assert isinstance(message.files[1], FailedFile)
    saved = restored(conversation).messages[-1]
    assert saved.files[1].reference == citation("bad", "archive.zip")
    assert "temporary download failure" in saved.files[1].error
    retrieve.side_effect = None
    retrieve.return_value = NS(content=b"recovered zip")
    if asynchronous:
        asyncio.run(adapter.retry_failed_files_async(saved))
    else:
        adapter.retry_failed_files(saved)
    assert retrieve.call_count == 3
    assert saved.files[0].data == b"script bytes"
    assert saved.files[1].data == b"recovered zip"
    assert saved.replay_data("openai", "test")["output"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_intermediate_tool_round_survives_download_failure(asynchronous):
    adapter = OpenAIAdapter()
    adapter.model_config = {"test": {"background_mode": False}}
    adapter._client = MagicMock()
    adapter._async_client = MagicMock()
    annotation = NS(type="container_file_citation", **citation("bad", "archive.zip"))
    first = NS(id="first", model="test", usage=None, status="completed", output=[
        NS(type="message", role="assistant", content=[NS(type="output_text", text="Working", annotations=[annotation])]),
        NS(type="function_call", id="fc", call_id="call", name="lookup", arguments="{}"),
    ])
    last = NS(id="last", model="test", usage=None, status="completed", output=[])

    def lookup():
        return {"value": 42}

    conversation = Conversation([Message("user", "export")])
    if asynchronous:
        adapter._async_client.responses.create = AsyncMock(side_effect=[first, last])
        adapter._async_client.containers.files.content.retrieve = AsyncMock(side_effect=OSError("unavailable"))
        message = asyncio.run(adapter.request_llm_async("test", conversation, functions=[lookup]))
    else:
        adapter._client.responses.create.side_effect = [first, last]
        adapter._client.containers.files.content.retrieve.side_effect = OSError("unavailable")
        message = adapter.request_llm("test", conversation, functions=[lookup])
    assert message.id == "last"
    assert isinstance(conversation.messages[1].files[0], FailedFile)
    assert conversation.messages[2].function_responses[0].response == {"value": 42}


def test_openai_sdk_serializes_multimodal_tool_results():
    import httpx
    from openai import OpenAI

    result = FunctionResponse("export", {"files": [png(), PDFDocumentFile(b"pdf", "report.pdf")]}, call_id="call")
    captured = []

    def send(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp", "object": "response", "created_at": 0, "model": "test",
            "status": "completed", "output": [], "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [],
        })

    with OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(send))) as client:
        client.responses.create(model="test", input=[function_response_to_openai(result)])
    parts = captured[0]["input"][0]["output"]
    assert [part["type"] for part in parts] == ["input_text", "input_image", "input_file"]
    assert parts[2]["file_data"] == "data:application/pdf;base64,cGRm"


def test_generated_jpeg_is_not_named_png():
    file = ImageFile.from_pil_image(Image.new("RGB", (2, 2)), "source.jpg")
    response = NS(model="test", usage=None, output=[NS(
        type="image_generation_call", result=file.base64, output_format="jpeg",
    )])
    generated = OpenAIAdapter()._parse_response(response)[2][0]
    assert generated.name.endswith(".jpeg")
    assert generated.data == file.data
    assert generated.mime_type == "image/jpeg"


@pytest.mark.parametrize("cls", [PDFDocumentFile, WordDocumentFile, PowerPointDocumentFile])
def test_corrupt_documents_raise_instead_of_becoming_empty_text(cls):
    with pytest.raises(DocumentExtractionError):
        _ = cls(b"corrupt", "document").text


def test_corrupt_pdf_page_count_is_not_zero():
    with pytest.raises(DocumentExtractionError):
        _ = PDFDocumentFile(b"corrupt", "test.pdf").number_of_pages


def test_scanned_pdf_is_native_when_possible_and_text_fallback_fails():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    writer.write(stream)
    file = PDFDocumentFile(stream.getvalue(), "scanned.pdf")
    assert OpenAIAdapter()._convert_file_to_content(file)["type"] == "input_file"
    with pytest.raises(DocumentExtractionError):
        _ = file.text


def test_partially_scanned_pdf_cannot_silently_drop_a_page(monkeypatch):
    reader = NS(pages=[NS(extract_text=lambda: "page one"), NS(extract_text=lambda: None)])
    monkeypatch.setattr("llm_platform.services.files.PdfReader", lambda _: reader)
    with pytest.raises(DocumentExtractionError, match="OCR"):
        _ = PDFDocumentFile(b"pdf", "mixed.pdf").text


def test_successful_text_extraction_warns_about_loss_and_empty_doc_is_distinct():
    for text in ["hello", ""]:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                             f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
        with pytest.warns(DocumentExtractionWarning, match="non-text content"):
            assert WordDocumentFile(stream.getvalue(), "document.docx").text == text
