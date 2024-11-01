FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv
RUN pipenv install --deploy --ignore-pipfile

COPY ["model1.bin ./", " dv.bin ./","app.py ./"]

EXPOSE 9696

ENTRYPOINT  ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "app:app"]