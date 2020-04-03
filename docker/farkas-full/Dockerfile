FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    git \
    make \
    graphviz

ARG DEPLOY_TOKEN_PWD

RUN git clone https://gitlab+deploy-token-1:$DEPLOY_TOKEN_PWD@gitlab.tcs.inf.tu-dresden.de/s3754766/farkas-tool farkas-tool

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
        
WORKDIR farkas-tool
RUN python setup.py build && python setup.py install

WORKDIR docs
RUN make html

WORKDIR /farkas-tool/examples
CMD ["jupyter","notebook","--ip=0.0.0.0","--no-browser","--allow-root","--NotebookApp.token=''"]