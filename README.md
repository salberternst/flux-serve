# flux serve

A small fastapi based service to run the FLUX.1 based models from [black forest labs](https://huggingface.co/black-forest-labs)

## Settings

The application can be configured using environment variables

| Name           | Description                                    | Default                      |
|----------------|------------------------------------------------|------------------------------|
| APP_MODEL_NAME | The name or path of the model | black-forest-labs/FLUX.1-schnell |
| APP_DEVICE     | The hardware device used for model inference   | cuda                         |
| APP_DTYPE      | The data type used for model computations      | float32                      |


## Run

Using docker: 

```shell
docker run \
    -p 8000 \
    -e APP_DEVICE=cpu \
    -e HF_TOKEN=xxx \
    -v $PWD/hub:/opt/huggingface \
    ghcr.io/salberternst/flux-serve:0.1.8
```

The container sets by default `HF_HOME` to `/opt/huggingface` to persit downloaded models.

## License

[MIT](./LICENSE)