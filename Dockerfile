# syntax=docker/dockerfile:1.4

FROM library/python:3.10.13-slim                                                                                                                                       

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update && apt-get install -y --no-install-recommends python3-opencv                                                  
RUN apt-get clean > /dev/null && rm -rf /var/lib/apt/lists/*                                                                                                           
COPY requirements.txt /tmp/requirements.txt                                                                                                     
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked pip install --upgrade pip && pip install -r /tmp/requirements.txt && \                                                                
    rm -f /tmp/requirements.txt 
RUN mkdir -p /tungsten/models                                                                                                                                       
COPY models/ /tungsten/models/