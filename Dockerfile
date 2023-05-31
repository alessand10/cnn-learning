FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:20230518.v1@sha256:8b905ebd2962666c164496d6ff5672cfce26b65149c64b6a4801213b01a91ef8
RUN apt-get update && apt-get install libgl1