FROM openjdk:11.0.7-slim-buster as prism-setup

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    make
    

RUN git clone https://github.com/prismmodelchecker/prism.git prism

WORKDIR prism/prism
RUN git checkout 6c21696ac8cbb402327eeaa956f282bcdb0eebb5
RUN make

FROM python:3.8-slim-buster as farkas-tool-setup

RUN apt-get update && apt-get install -y \
    git

ARG DEPLOY_TOKEN_PWD

RUN git clone https://github.com/prismmodelchecker/prism-benchmarks.git prism-benchmarks

RUN git clone https://gitlab+deploy-token-1:$DEPLOY_TOKEN_PWD@gitlab.tcs.inf.tu-dresden.de/s3754766/farkas-tool farkas-tool

WORKDIR farkas-tool
RUN python setup.py build

FROM python:3.8-slim-buster

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get install -y \
    default-jre-headless

COPY --from=farkas-tool-setup /farkas-tool /farkas-tool
COPY --from=farkas-tool-setup /prism-benchmarks /prism-benchmarks

COPY --from=prism-setup /prism /prism
ENV PATH="/prism/prism/bin:${PATH}"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY experiments.py experiments.py
CMD ["python","experiments.py"]
