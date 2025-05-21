from .adapter_base import AdapterBase
from llm_platform.tools.base import BaseTool
from llm_platform.services.files import MediaFile, ImageFile, AudioFile, VideoFile, TextDocumentFile, ExcelDocumentFile, PDFDocumentFile
from llm_platform.services.conversation import Conversation, FunctionCall, FunctionResponse, Message

from google import genai
from google.protobuf import struct_pb2
from google.genai import types

import time
from typing import List, Tuple, Callable, Dict
import os
import logging
from io import BytesIO
import asyncio
import uuid
import json

class GoogleAdapter(AdapterBase):
    
    def __init__(self, logging_level=logging.INFO):
        super().__init__(logging_level)   
        self.client = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'),
                                   http_options={'api_version': 'v1alpha'})

    def convert_conversation_history_to_adapter_format(self, the_conversation: Conversation):
        history = []

        # Add history of messages
        for message in the_conversation.messages:
            GEMINI_ROLE_MAPPING = {
                'user': 'user',
                'assistant': 'model',
                #'function': 'function'
            }
            try:
                role = GEMINI_ROLE_MAPPING[message.role]
            except KeyError:
                raise ValueError(f"Invalid role in history: {message.role}")
            
            # For role "function":
            if message.function_calls or message.function_responses:
                # Convert function calls to history format
                function_calls_parts = []
                for each_function_call in message.function_calls:
                    struct_arguments = struct_pb2.Struct()
                    struct_arguments.update(json.loads(each_function_call.arguments))
                    function_call_part = genai.types.Part.from_function_call(name=each_function_call.name, args=struct_arguments)

                    function_calls_parts.append(function_call_part)

                # Convert function responses to history format
                response_parts = []
                for each_function_response in message.function_responses:
                    #if isinstance(each_function_response.response, list): 
                        # perhaps you want a dict with "value"? 
                    #    payload = {"items": each_function_response.response} 
                    #else: 
                    #    payload = each_function_response.response
                    
                    response_part = genai.types.Part.from_function_response(name=each_function_response.name, 
                                                                      response=each_function_response.response,
                                                                    )
                    response_parts.append(response_part)
                    
                    protos_message = genai.types.Content(role="function", 
                                                        parts = function_calls_parts + response_parts)
                    
                    history.append(protos_message)
                    continue
            
            message_parts = []
            
            # Add text part
            if message.content:
                message_parts.append(genai.types.Part.from_text(text=message.content))
                #protos_message = genai.types.Content(role=role, parts=[genai.types.Part.from_text(text=message.content)])

            # Add files to history (for the moment, only images)
            if not message.files is None:
                for each_file in message.files:
                    
                    # Images
                    if isinstance(each_file, ImageFile):
                        #print(f"File name: {each_file.name}, file extension: {each_file.extension}")
                        part_with_image = genai.types.Part.from_bytes(data = each_file.file_bytes,
                                                                mime_type = f"image/{each_file.extension}")

                        message_parts.append(part_with_image)

                    elif isinstance(each_file, AudioFile):
                        part_with_audio = genai.types.Part.from_bytes(data = each_file.file_bytes,
                                                                mime_type = f"audio/mp3")

                        message_parts.append(part_with_audio)

                    # Text documents
                    elif isinstance(each_file, (TextDocumentFile, ExcelDocumentFile)):
                        document_as_text = f"""<document name="{each_file.name}">{each_file.text}</document>"""
                        part_with_text_document = genai.types.Part.from_text(text=document_as_text)

                        message_parts.append(part_with_text_document)

                    # PDF documents
                    elif isinstance(each_file, PDFDocumentFile):
                        if (each_file.size < 20_000_000) and (each_file.number_of_pages < 3_600):
                            part_with_pdf = genai.types.Part.from_bytes(data = each_file.bytes,
                                                                mime_type = f"application/pdf")

                            message_parts.append(part_with_pdf)
                        else: # Load pdf as text
                            document_as_text = f"""<document name="{each_file.name}">{each_file.text}</document>"""
                            part_with_text_document = genai.types.Part.from_text(text=document_as_text)

                            message_parts.append(part_with_text_document)


            protos_message = genai.types.Content(role=role, parts=message_parts)
            history.append(protos_message)

        return history

    def request_llm(self, model: str,
                    the_conversation: Conversation, 
                    functions:List[Callable]=None, 
                    temperature: int=0,  
                    tool_output_callback: Callable=None,
                    additional_parameters: Dict={},
                    **kwargs) -> Message:
        
        # Modify kwargs to change max_tokens to max_output_tokens
        if 'max_tokens' in kwargs:
            kwargs['max_output_tokens'] = kwargs.pop('max_tokens')
        else:
            # Fetch max_tokens from the model config
            max_tokens = the_conversation.model_config.get_max_tokens(model)
            kwargs['max_output_tokens'] = max_tokens

        if functions is None:

            history = self.convert_conversation_history_to_adapter_format(the_conversation)

            # Prepare parameters for the generation config
            generation_config_params = {
                "system_instruction": the_conversation.system_prompt if the_conversation.system_prompt else "",
                "temperature": temperature,
                "tools": [],
                "safety_settings": self.safety_settings,
            }

            # Convert parameter `reasoning_efforts` to `thinking` parameter
            if 'reasoning' in kwargs:
                reasoning_effort = kwargs.pop('reasoning', {}).get('effort', 'low')
                """
                    reasoning_effort 'low' -> thinking is not used at all
                    reasoning_effort 'medium' -> thinking budget is 8000
                    reasoning_effort 'high' -> thinking budget is 32000
                """
                if reasoning_effort in ['high', 'medium']:
                    reasoning_effort_map = {'high': 24_576, 'medium': 8_000}
                    thinking_budget = reasoning_effort_map[reasoning_effort]
                    generation_config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

            # Add response modalities to the generation config parameters
            if "response_modalities" in additional_parameters:
                generation_config_params["response_modalities"] = additional_parameters["response_modalities"]

            generation_config_params.update(kwargs)

            generation_config = genai.types.GenerateContentConfig(**generation_config_params)

            # Grounding
            if additional_parameters.get("grounding", False):
                grounding_tool = genai.types.Tool(google_search=genai.types.GoogleSearchRetrieval)
                generation_config.tools.append(grounding_tool)

            # URL context
            if additional_parameters.get("url_context", False):
                url_context_tool = genai.types.Tool(url_context=genai.types.UrlContext)
                generation_config.tools.append(url_context_tool)

            response = self.client.models.generate_content(contents = history, 
                                                      model = model,
                                                      config = generation_config,
                                                    )
        else:
            response = self.request_llm_with_functions(model, 
                                                       the_conversation, 
                                                       functions, 
                                                       temperature,
                                                       tool_output_callback,
                                                       **kwargs)

        finish_reason = response.candidates[0].finish_reason
        safety_ratings = response.candidates[0].safety_ratings

        # Parse response
        text_from_response = ""
        files_from_response: List[MediaFile] = []

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text_from_response += part.text
            elif part.inline_data is not None:
                if part.inline_data.mime_type == "image/png":
                    image = ImageFile.from_bytes(file_bytes=part.inline_data.data, 
                                                 file_name=f"image_{len(files_from_response)}.png")
                    files_from_response.append(image)
                else:
                    raise ValueError(f"Unsupported mime type: {part.inline_data.mime_type}")

        # Get usage
        usage = {"model": model,
                "completion_tokens": response.usage_metadata.prompt_token_count,
                "prompt_tokens": response.usage_metadata.candidates_token_count}
        
        message = Message(role="assistant", content=text_from_response, files=files_from_response, usage=usage)
        the_conversation.messages.append(message)

        return message
    
    @property
    def safety_settings(self):
        # Safety config
        safety_settings = [
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        return safety_settings

    def request_llm_with_functions(self,
                                   model: str, 
                                   the_conversation: Conversation, 
                                   functions: List[BaseTool]=[], 
                                   temperature: int=0, 
                                   tool_output_callback: Callable=None,
                                   additional_parameters: Dict={},
                                   **kwargs
                                   ): 
        # Convert all functions by letting your BaseTool clean the schema 
        converted_functions = [] 
        for func in functions: 
                if isinstance(func, BaseTool): 
                    # This will remove 'title' from the schema 
                    function_decls = func.to_params(provider="google") 
                    
                    # Wrap in "function_declarations" as the gemini client expects 
                    converted_functions.append({"function_declarations": [function_decls]}) 
                else: 
                    # If it’s not a llm_platform.tools.base.BaseTool 
                    converted_functions.append(func)

        # Prepare parameters for the generation config
        generation_config_params = {
            "system_instruction": the_conversation.system_prompt if the_conversation.system_prompt else "",
            "temperature": temperature,
            "tools": converted_functions,
            "safety_settings": self.safety_settings,
        }

        # Convert parameter `reasoning_efforts` to `thinking` parameter
        if 'reasoning' in kwargs:
            reasoning_effort = kwargs.pop('reasoning', {}).get('effort', 'low')
            """
                reasoning_effort 'low' -> thinking is not used at all
                reasoning_effort 'medium' -> thinking budget is 8000
                reasoning_effort 'high' -> thinking budget is 32000
            """
            if reasoning_effort in ['high', 'medium']:
                reasoning_effort_map = {'high': 24_576, 'medium': 8_000}
                thinking_budget = reasoning_effort_map[reasoning_effort]
                generation_config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        # Add response modalities to the generation config parameters
        if "response_modalities" in additional_parameters:
            generation_config_params["response_modalities"] = additional_parameters["response_modalities"]

        generation_config_params.update(kwargs)

        generation_config = genai.types.GenerateContentConfig(**generation_config_params)

        # Grounding
        if additional_parameters.get("grounding", False):
            grounding_tool = genai.types.Tool(google_search=genai.types.GoogleSearchRetrieval)
            generation_config.tools.append(grounding_tool)

        while True:
        
            history = self.convert_conversation_history_to_adapter_format(the_conversation)

            response = self.client.models.generate_content(contents = history, 
                                                      model = model,
                                                      config = generation_config,
                                                    )
            
            usage = {"model": model,
                "completion_tokens": response.usage_metadata.prompt_token_count,
                "prompt_tokens": response.usage_metadata.candidates_token_count
            }
            
            function_calls = []
            function_responses = []
            # Iterate through all function calls in the response
            for part in response.candidates[0].content.parts:

                # Call each requested function
                if function_call := part.function_call:
                    # Get function name and arguments
                    function_id = str(uuid.uuid4())
                    function_args = dict(function_call.args)
                    function_name = function_call.name

                    # Save record of the function call
                    function_call_record = FunctionCall(id=function_id,
                                                        name=function_name,
                                                        arguments=json.dumps(function_args))
                    function_calls.append(function_call_record)

                    # Find the requested function
                    function = next((each_function for each_function in functions if each_function.__name__ == function_name), None)
                    if function is None:
                        raise ValueError(f"Function {function_name} not found in tools")

                    # Call the function
                    # IMPORTANT!: function response should be Dict
                    function_response = function(**function_args)

                    #logging.debug(f"Function response: {function_response}")
                    response_struct = FunctionResponse(name=function_name,
                                                        response=function_response,
                                                        id = function_id
                                                        )
                    function_responses.append(response_struct)

                    if tool_output_callback:
                        tool_output_callback(function_name,
                                            function_args,
                                            function_response
                                            )

            if len(function_calls) == 0:
                message = Message(role="assistant", 
                                    content=response.candidates[0].content.parts[0].text,
                                    usage=usage
                                    )
                the_conversation.messages.append(message)

                return response

            # Save function calls and function response to the history
            message = Message(role="assistant", 
                                        content=" ",
                                        function_calls=function_calls,
                                        function_responses=function_responses,
                                        usage=usage
                                        )
            the_conversation.messages.append(message)

    def get_models(self) -> List[str]:
        NotImplementedError("Not implemented yet")

    def generate_image(self, prompt: str, n: int=1, **kwargs) -> List[ImageFile]: 
        """
        Parameters:
            prompt: The text prompt for the image.
            number_of_images: The number of images to generate, from 1 to 4 (inclusive). The default is 4.
            aspect_ratio: Changes the aspect ratio of the generated image. 
                Supported values are "1:1", "3:4", "4:3", "9:16", and "16:9". The default is "1:1".
            safety_filter_level: Adds a filter level to safety filtering. The following values are valid:
                "BLOCK_LOW_AND_ABOVE": Block when the probability score or the severity score is LOW, MEDIUM, or HIGH.
                "BLOCK_MEDIUM_AND_ABOVE": Block when the probability score or the severity score is MEDIUM or HIGH.
                "BLOCK_ONLY_HIGH": Block when the probability score or the severity score is HIGH.
            person_generation: Allow the model to generate images of people. The following values are supported:
                "DONT_ALLOW": Block generation of images of people.
                "ALLOW_ADULT": Generate images of adults, but not children. This is the default.
        """

        """
            # Examples of the generation_config
            generation_config=genai.types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                max_output_tokens=100,
                stop_sequences=['STOP!'],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            ),
        """

        response = self.client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(
                number_of_images=n,
                **kwargs
            ),
        )

        images = [ImageFile.from_bytes(file_bytes=generated_image.image.image_bytes, file_name="image.webp") for generated_image in response.generated_images]
            
        return images
    
    async def generate_image_async(self, prompt: str, n: int=1, **kwargs) -> List[BytesIO]: 
        """
        Parameters:
            prompt: The text prompt for the image.
            number_of_images: The number of images to generate, from 1 to 4 (inclusive). The default is 4.
            aspect_ratio: Changes the aspect ratio of the generated image. 
                Supported values are "1:1", "3:4", "4:3", "9:16", and "16:9". The default is "1:1".
            safety_filter_level: Adds a filter level to safety filtering. The following values are valid:
                "BLOCK_LOW_AND_ABOVE": Block when the probability score or the severity score is LOW, MEDIUM, or HIGH.
                "BLOCK_MEDIUM_AND_ABOVE": Block when the probability score or the severity score is MEDIUM or HIGH.
                "BLOCK_ONLY_HIGH": Block when the probability score or the severity score is HIGH.
            person_generation: Allow the model to generate images of people. The following values are supported:
                "DONT_ALLOW": Block generation of images of people.
                "ALLOW_ADULT": Generate images of adults, but not children. This is the default.
        """

        """
            # Examples of the generation_config
            generation_config=genai.types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                max_output_tokens=100,
                stop_sequences=['STOP!'],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            ),
        """

        response = self.client.aio.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(
                number_of_images=n,
                **kwargs
            )
        )

        images = [BytesIO(generated_image.image.image_bytes) for generated_image in response.generated_images]
            
        return images

    def generate_video(self, 
                       prompt: str, 
                       aspect_ratio: str="16:9", 
                       person_generation: str="ALLOW_ADULT", 
                       image: ImageFile=None, 
                       number_of_videos: int=1,
                       negative_prompt: str=None,
                       duration_seconds: int=5,
                    ):
        """
        Parameters:
            prompt: str The text prompt for the image.
            aspect_ratio: str Changes the aspect ratio of the generated image. 
                Supported values are "9:16", and "16:9".
            person_generation: str Allow the model to generate images of people. The following values are supported:
                "DONT_ALLOW": Block generation of images of people.
                "ALLOW_ADULT": Generate images of adults, but not children. This is the default.
            image: ImageFile The image to use as a reference for the video generation.
            number_of_videos: int The number of videos to generate. The default is 1.
            negative_prompt: str The text prompt for the image.
            duration_seconds: int The duration of the video in seconds (from 5 to 8). The default is 5 seconds.
        """
        
        parameters = {"model": "veo-2.0-generate-001",
                      "prompt": prompt,
                    }
        
        if image:
            parameters["image"] = genai.types.Image(image=image.file_bytes, mime_type=f"image/{image.extension}")
            person_generation = "DONT_ALLOW" # person_generation only accepts "dont_allow" for image-to-video

        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt

        config = types.GenerateVideosConfig(
                person_generation=person_generation,  # "dont_allow" or "allow_adult"
                aspect_ratio=aspect_ratio,  # "16:9" or "9:16"
                number_of_videos=number_of_videos,  # Number of videos to generate
        )

        if duration_seconds:
            config.duration_seconds = duration_seconds

        parameters["config"] = config

        operation = self.client.models.generate_videos(**parameters)

        while not operation.done:
            time.sleep(20)
            operation = self.client.operations.get(operation)

        for n, generated_video in enumerate(operation.response.generated_videos):
            self.client.files.download(file=generated_video.video)
            generated_video.video.save(f"video{n}.mp4")  # save the video