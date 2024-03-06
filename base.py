import os
import json
import asyncio
from typing import Dict, Any
import aiohttp
from dotenv import load_dotenv


class Config:
    load_dotenv()
    AUTH_TOKEN = os.environ["REPLICATE_AUTH_TOKEN"]
    BASE_URL = "https://api.replicate.com/v1"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://replicate.com/stability-ai/sdxl",
        "Content-Type": "application/json",
        "Origin": "https://replicate.com",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "TE": "trailers",
    }


class ReplicateClient:
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.headers = Config.HEADERS

    async def get(self, endpoint: str, auth_required: bool = False) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        headers = self.headers.copy()
        if auth_required:
            headers["Authorization"] = f"Token {Config.AUTH_TOKEN}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                return await response.json()

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as response:
                return await response.json()


class CollectionRepository:
    def __init__(self, client: ReplicateClient):
        self.client = client

    async def fetch_collections(self) -> Dict[str, Any]:
        return await self.client.get("collections", auth_required=True)

    async def fetch_models(self, collection_slug: str) -> Dict[str, Any]:
        return await self.client.get(
            f"collections/{collection_slug}", auth_required=True
        )


class ModelRepository:
    def __init__(self, client: ReplicateClient):
        self.client = client

    async def send_prediction(
        self, model_id: str, input_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        data = {
            "input": input_params,
            "is_training": False,
            "create_model": "0",
            "stream": True,
            "version": model_id,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://replicate.com/api/predictions", json=data
            ) as response:
                response_json = await response.json()

        prediction_id = response_json["id"]
        print(f"Prediction id: {prediction_id}")

        while response_json["status"] != "succeeded":
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://replicate.com/api/predictions/{prediction_id}"
                ) as response:
                    response_json = await response.json()
            await asyncio.sleep(1)  # Wait for 1 second before checking again

        return response_json


class ReplicateAPI:
    def __init__(self):
        self.client = ReplicateClient()
        self.collection_repo = CollectionRepository(self.client)
        self.model_repo = ModelRepository(self.client)
        self.collections: Dict[str, str] = {}
        self.models: Dict[str, Dict[str, Any]] = {}

    async def fetch_data(self) -> None:
        collections_data = await self.collection_repo.fetch_collections()
        for collection in collections_data["results"]:
            slug = collection["slug"]
            description = collection["description"]
            self.collections[slug] = description

            models_data = await self.collection_repo.fetch_models(slug)
            for model in models_data["models"]:
                name = model["name"]
                latest_version = model.get("latest_version")
                if latest_version:
                    model_id = latest_version["id"]
                    default_input = model["default_example"]["input"]
                    category = slug
                    description = model["description"]
                    self.models[name] = {
                        "id": model_id,
                        "default_input": default_input,
                        "category": category,
                        "description": description,
                    }

    def get_collections(self) -> Dict[str, str]:
        return self.collections

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return self.models

    async def send_prediction(
        self, model_name: str, input_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        model_id = self.models[model_name]["id"]
        return await self.model_repo.send_prediction(model_id, input_params)

    async def save_collections_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(self.collections, file, indent=4)

    async def save_models_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(self.models, file, indent=4)


async def main():
    api = ReplicateAPI()
    # await api.fetch_data()
    # print(api.get_collections())
    # print(api.get_models())

    # model_name = "stability-ai/sdxl"
    input_params = {
        "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic"
    }
    prediction = await api.send_prediction("llama-2-13b-gguf", input_params)
    print(prediction)

    # await api.save_collections_to_json("collections.json")
    # await api.save_models_to_json("models.json")
    #


if __name__ == "__main__":
    asyncio.run(main())
