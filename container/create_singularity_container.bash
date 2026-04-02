
# Build a Docker image and convert to a Singularity container for HPC clusters.
NAME=anvil-audio
docker build -t ${NAME} -f ./container/${NAME}.Dockerfile .

# Convert to Singularity
singularity build ${NAME}.sif docker-daemon://${NAME}
