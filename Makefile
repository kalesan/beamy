all:
	cd docker/frontend && ./build.sh
	cd docker/simulation-server && ./build.sh
