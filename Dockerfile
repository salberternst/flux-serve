FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /flux_serve

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /flux_serve/app

USER 1000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0"]

EXPOSE 8000