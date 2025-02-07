# python_template

## Environment

Our experiment was conducted on Ubuntu with an NVIDIA A100 Tensor Core GPU. If
you want to re-train model, we reccomend to use GPU.

## Installation using Docker

```shell
git clone git@github.com:inoue0426/DrugAgent2.git
cd DrugAgent2
docker build -t tmp:latest .
docker run -it -p 9999:9999 -v $(pwd):/app tmp:latest
conda activate python_env
```

At the docker env.

```shell
cd agents/ & python integrate_agent.py
```
