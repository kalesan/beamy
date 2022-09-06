#!/bin/bash

docker build \
       --build-arg "GRAFANA_VERSION=9.0.6" \
       --build-arg "GF_INSTALL_PLUGINS=marcusolsson-csv-datasource" \
       -t beamy-frontend -f Dockerfile .
