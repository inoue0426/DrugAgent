FROM continuumio/miniconda3:23.10.0

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl=7.88.1-10+deb12u5 \
    wget=1.21.3-1+b1 \
    jq=1.6-2.1 \
    vim=2:9.0.1378-2 && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

SHELL ["/bin/bash", "-c"]
RUN source activate python_env && \
    pip install --no-cache-dir \
    git+https://github.com/bp-kelley/descriptastorus@master \
    "flaml[automl]==1.2.4"

COPY . /app

EXPOSE 9999

CMD ["/bin/bash", "-c", "source activate python_env && cd agents && nohup jupyter lab --ip=0.0.0.0 --port=9999 --no-browser --allow-root --NotebookApp.token='' > jupyter.log 2>&1 & bash"]
