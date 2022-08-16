#!/bin/bash

docker build \
       --build-arg "GRAFANA_VERSION=9.0.6" \
       --build-arg "GF_INSTALL_PLUGINS=https://github.com/grafana/mqtt-datasource/archive/refs/heads/main.zip;grafana-simple-json-datasource,natel-plotly-panel,ae3e-plotly-panel" \
       -t grafana-beamy -f Dockerfile .
