FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pip install pipenv
RUN pipenv install --deploy --ignore-pipfile

COPY app.py ./
COPY model1.bin ./
COPY dv.bin ./

EXPOSE 9696

CMD ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "app:app"]