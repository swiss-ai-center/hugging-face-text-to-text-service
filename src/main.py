import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import requests

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
                    name="result", type=[FieldDescriptionType.APPLICATION_JSON,
                                         FieldDescriptionType.TEXT_PLAIN]
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
                output_list = [data[desired_output] for data in result_data.json() if desired_output
                               in data]
                # This list should contain only one element, the desired output field from the received JSON
                output = output_list[0]
            else:
                output = result_data.json()[desired_output]

            return {
                "result": TaskData(data=output,
                                   type=FieldDescriptionType.TEXT_PLAIN)
            }

        return {
            "result": TaskData(data=output,
                               type=FieldDescriptionType.APPLICATION_JSON)
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(my_service, engine_url)
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


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

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Hugging Face text-to-text service",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
