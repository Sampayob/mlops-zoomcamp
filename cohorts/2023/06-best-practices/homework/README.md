# Docker-compose commands
> docker-compose -f docker-compose.yml up

# AWS CLI commands

## Make bucket
> aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

## List bucket files with size
> aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration --recursive --summarize 