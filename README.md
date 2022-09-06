# Beamy 

Beamy is an open source physical layer general purpose simulator for wireless systems. 

## Frontend && Visualization 
Beamy integrages with [https://grafana.org](Grafana.com) to provide modern frontend for visualization and investigation of the simulation results.

### Requirements
- Working docker environment
- docker-compose

### Build frontend
```
make
```

### Start frontend

Start frontend on default address (http://localhost:3000) 
```
./frontend.sh 
```

Use different port the frontend access (http://localhost:3004)
```
PORT=3004 ./frontend.sh
```

Serve simulation results from another path
```
SIMS_PATH=${PWD}/another-simulation-path ./frontend.sh
```

# Purpose
It is intended for research and education use.
