# flux serve

A simple FastAPI-based service to run the FLUX.1 based models from [black forest labs](https://huggingface.co/black-forest-labs)

## Settings

The application can be configured using environment variables

| Name           | Description                                    | Default                      |
|----------------|------------------------------------------------|------------------------------|
| APP_MODEL_NAME | The name or path of the model | black-forest-labs/FLUX.1-schnell |
| APP_DEVICE     | The hardware device used for model inference   | cuda                         |
| APP_DTYPE      | The data type used for model computations      | float32                      |


## Run

Using docker: 

```bash
docker run \
    -p 8000 \
    -e APP_DEVICE=cpu \
    -e HF_TOKEN=your_hf_token \
    -v $PWD/hub:/opt/huggingface \
    ghcr.io/salberternst/flux-serve:0.1.8
```

Note: The container sets HF_HOME to /opt/huggingface to persist downloaded models.

Using the [helm chart](./charts/flux-serve/README.md):

```bash
helm repo add flux-serve https://salberternst.github.io/flux-serve
helm install flux-serve flux-serve/flux-serve \
    --set flux_serve.device=cpu
```

## License

[MIT](./LICENSE)