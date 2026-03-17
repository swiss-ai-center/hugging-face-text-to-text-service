from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import requests
import json

api_description = """The service is used to query text-to-image AI models from the Hugging Face inference API.\n

You can choose from any model available on the inference API from the [Hugging Face Hub](https://huggingface.co/models)
that takes a text(json) as input and outputs text(json).

It must take only one json input with the following structure:

```
{
    "inputs" : "your input text"
}
```

This service has two input files:
 - A json file that defines the model you want to use, your access token and optionally a desired output field from the
  json answer.
 - A text file used as input.

json_description.json example:
 ```
 {
    "api_token": "your_token",
    "api_url": "https://api-inference.huggingface.co/models/gpt2",
    "desired_output": "generated_text"
}
```
This model, "gpt2", is used for text generation.

!!! note

    If you don't specify a desired output, the service will return the whole JSON file (.json).
    If you do specify an output, the response will be a text file containing the given field data.

The model may need some time to load on Hugging face's side, you may encounter an error on your first try.

Helpful trick: The answer from the inference API is cached, so if you encounter a loading error try to change the
input to check if the model is loaded.
"""

api_summary = """This service is used to query text-to-text models from Hugging Face
"""
api_title = "Hugging Face text-to-text service"
version = "1.0.0"

settings = get_settings()


class MyService(Service):
    """
    Hugging Face service uses Hugging Face's model hub API to directly query text-to-text AI models
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Hugging Face text-to-text",
            slug="hugging-face-text-to-text",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(
                    name="json_description",
                    type=[
                        FieldDescriptionType.APPLICATION_JSON
                    ],
                ),
                FieldDescription(
                    name="input_text",
                    type=[
                        FieldDescriptionType.TEXT_PLAIN
                    ]
                ),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.NATURAL_LANGUAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.NATURAL_LANGUAGE_PROCESSING,
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/hugging-face-text-to-text/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        def is_valid_json(json_string):
            try:
                json.loads(json_string)
                return True
            except ValueError:
                return False

        try:
            json_description = json.loads(data['json_description'].data.decode('utf-8'))
            api_token = json_description['api_token']
            api_url = json_description['api_url']
        except ValueError as err:
            raise Exception(f"json_description is invalid: {str(err)}")
        except KeyError as err:
            raise Exception(f"api_url or api_token missing from json_description: {str(err)}")

        headers = {"Authorization": f"Bearer {api_token}"}

        def natural_language_query(payload):
            response = requests.post(api_url, headers=headers, json=payload)
            return response

        def flatten_list(lst):
            flattened_list = []
            for item in lst:
                if isinstance(item, list):
                    flattened_list.extend(item)
                else:
                    flattened_list.append(item)
            return flattened_list

        input_text_bytes = data['input_text'].data
        json_input_text = f'{{ "inputs" : "{input_text_bytes.decode("utf-8")}" }}'
        json_payload = json.loads(json_input_text)
        result_data = natural_language_query(json_payload)

        if is_valid_json(result_data.content):
            data = json.loads(result_data.content)
            if 'error' in data:
                raise Exception(data['error'])

        output = json.dumps(result_data.json(), indent=4)

        if 'desired_output' in json_description:
            desired_output = json_description['desired_output']
            if isinstance(result_data.json(), list):
                flat_list = flatten_list(result_data.json())
                # If several objects contain the desired output, return all the objects with that field only.
                output_list = [{desired_output: data[desired_output]} for data in flat_list if desired_output
                               in data]
                output = json.dumps(output_list, indent=4)
            else:
                output = result_data.json()[desired_output]

        return {
            "result": TaskData(data=output,
                               type=FieldDescriptionType.APPLICATION_JSON)
        }
