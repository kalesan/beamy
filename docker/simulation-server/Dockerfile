FROM alpine:latest

RUN apk add --update --no-cache py3-numpy py3-pandas

COPY ./serve_csv.py /

ENTRYPOINT ["python3", "/serve_csv.py"]
