FROM python:3.12

RUN apt-get update && apt-get install -y libgl1-mesa-glx
WORKDIR /app
    
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD ["flask", "--app=app", "run", "--host=0.0.0.0"]