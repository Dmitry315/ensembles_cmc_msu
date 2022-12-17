FROM python:3.8-slim

COPY src/requirements.txt /root/src/requirements.txt

RUN chown -R root:root /root

WORKDIR /root/src
RUN pip3 install -r requirements.txt

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY 1515dd15dd3d5d1a51b5af515ca
ENV FLASK_APP flask_server

RUN chmod +x flask_server.py
CMD ["python3", "flask_server.py"]
