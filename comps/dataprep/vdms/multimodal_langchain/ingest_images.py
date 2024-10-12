# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from tqdm import tqdm
from utils import store_embeddings
from utils.utils import process_all_images, read_config
from utils.clip import CLIP
from utils.resnet import ResNet

from comps import opea_microservices, register_microservice, CustomLogger

VECTORDB_SERVICE_HOST_IP = os.getenv("VDMS_HOST", "0.0.0.0")
VECTORDB_SERVICE_PORT = os.getenv("VDMS_PORT", 55555)
collection_name = os.getenv("INDEX_NAME", "rag-vdms")
logger = CustomLogger("ingest_images")
logflag = os.getenv("LOGFLAG", False)


def setup_model(config, device="cpu"):
    is_transformer = int(config["embeddings"]["is_transformer"])

    if logflag:
        logger.info(f"[ setup_model ] is_transformer:{is_transformer}")

    if is_transformer:
        model = CLIP(config)
    else:
        # TODO:
        model = ResNet(config)

    return model


def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x


def store_into_vectordb(vs, metadata_file_path):
    GMetadata = read_json(metadata_file_path)

    total_images = len(GMetadata.keys())

    for idx, (image, data) in enumerate(tqdm(GMetadata.items())):
        metadata_list = []
        ids = []

        data["image"] = image
        image_name_list = [data["image_path"]]
        metadata_list = [data]
        if vs.selected_db == "vdms":
            vs.video_db.add_images(
                uris=image_name_list,
                metadatas=metadata_list
            )
        else:
            print(f"ERROR: selected_db {vs.selected_db} not supported. Supported:[vdms]")

    # clean up tmp_ folders containing frames (jpeg)
    for i in os.listdir():
        if i.startswith("tmp_"):
            print("removing tmp_*")
            os.system("rm -r tmp_*")
            break


def generate_image_id():
    """Generates a unique identifier for a image file."""
    return str(uuid.uuid4())


def generate_embeddings(config, vs):
    # process image(s) metadata and dump to metadata.json
    process_all_images(config)
    global_metadata_file_path = os.path.join(config["meta_output_dir"], "metadata.json")
    print(f"global metadata file available at {global_metadata_file_path}")
    store_into_vectordb(vs, global_metadata_file_path)


@register_microservice(name="opea_service@prepare_image_vdms", endpoint="/v1/dataprep", host="0.0.0.0", port=6007)
async def process_images(files: List[UploadFile] = File(None)):
    """Ingest images to VDMS."""

    config = read_config("./config-vision.yaml")
    path = config["images"]
    meta_output_dir = config["meta_output_dir"]
    emb_path = config["embeddings"]["path"]
    host = VECTORDB_SERVICE_HOST_IP
    port = int(VECTORDB_SERVICE_PORT)
    selected_db = config["vector_db"]["choice_of_db"]
    print(f"Parsing videos {path}.")

    # Saving images
    if files:
        image_files = []
        for file in files:
            file_trailing = os.path.splitext(file.filename)[1]
            if file_trailing == ".jpg" or file_trailing == ".png":
                image_files.append(file)
            else:
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not a png or jpg file. Please upload png and/or jpg files only."
                )

        for image_file in image_files:
            image_id = generate_image_id()
            image_info = os.path.splitext(image_file.filename)
            image_name = image_info[0]
            image_trailing = image_info[1]
            image_file_name = f"{image_name}_{image_id}{image_trailing}"
            image_dir_name = os.path.splitext(image_file_name)[0]
            # Save video file in upload_directory
            with open(os.path.join(path, image_file_name), "wb") as f:
                shutil.copyfileobj(image_file.file, f)

    # Creating DB
    print(
        "Creating DB with image embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time.",
        flush=True,
    )
    print("Connecting to {} at {}:{}".format(selected_db, host, port), flush=True)

    # init meanclip model
    model = setup_model(config, device="cpu")
    vs = store_embeddings.VideoVS(
        host, port, selected_db, model, collection_name
    )
    print("done creating DB, sleep 5s", flush=True)
    time.sleep(5)

    generate_embeddings(config, vs)

    return {"message": "Images ingested successfully"}


@register_microservice(
    name="opea_service@prepare_image_vdms",
    endpoint="/v1/dataprep/get_images",
    host="0.0.0.0",
    port=6007,
    methods=["GET"],
)
async def rag_get_file_structure():
    """Returns list of names of uploaded images saved on the server."""
    config = read_config("./config-vision.yaml")
    if not Path(config["images"]).exists():
        print("No file uploaded, return empty list.")
        return []

    uploaded_images = os.listdir(config["images"])
    image_files = [file for file in uploaded_images if file.endswith((".jpg", ".png"))]
    return image_files


@register_microservice(
    name="opea_service@prepare_image_vdms",
    endpoint="/v1/dataprep/get_file/{filename}",
    host="0.0.0.0",
    port=6007,
    methods=["GET"],
)
async def rag_get_file(filename: str):
    """Download the file from remote."""

    config = read_config("./config-vision.yaml")
    UPLOAD_DIR = config["images"]
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename)
    else:
        return {"error": "File not found"}


if __name__ == "__main__":
    opea_microservices["opea_service@prepare_image_vdms"].start()
