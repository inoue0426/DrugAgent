FROM continuumio/miniconda3

WORKDIR /app

ENV PATH /opt/conda/bin:$PATH

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

RUN /bin/bash -c "source activate python_env"

COPY . /app

EXPOSE 9999
CMD [ "/bin/bash", "-c", "source activate python_env && nohup jupyter notebook --ip=0.0.0.0 --port=9999 --no-browser --allow-root --NotebookApp.token='' > jupyter.log 2>&1 & bash" ]
