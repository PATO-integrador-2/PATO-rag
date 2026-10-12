FROM python:3.12-slim

WORKDIR /app

# RUN apt-get -y update
# RUN apt-get -y --no-install-recommends install libgomp1

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 12020

CMD [ "python", "./src/api/main.py" ]