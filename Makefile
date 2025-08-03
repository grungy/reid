# Makefile for managing Docker Compose

COMPOSE_FILE=docker-compose.yml

.PHONY: pgadmin
pgadmin:
	poetry run python generate_pgadmin_config.py
	docker cp servers.json i7y-dev-pgadmin-1:/pgadmin4/servers.json
	sudo docker exec -u root i7y-dev-pgadmin-1 chmod 600 /pgadmin4/servers.json
	sudo docker exec -u root i7y-dev-pgadmin-1 chown pgadmin:root /pgadmin4/servers.json
	docker restart i7y-dev-pgadmin-1
# Default target (optional)
.PHONY: up
up:
	docker compose -f $(COMPOSE_FILE) up --pull always -d

.PHONY: down
down:
	docker compose -f $(COMPOSE_FILE) down

.PHONY: cleanImages
cleanImages:
	# Stops containers and removes all images created by the Compose file
	docker compose -f $(COMPOSE_FILE) down --rmi all

.PHONY: prune
prune:
	# Removes all unused Docker images, containers, and networks
	docker system prune -af

.PHONY: rebuild
rebuild:
	# Rebuilds all images and starts containers
	docker compose -f $(COMPOSE_FILE) up --build -d

.PHONY: worker
worker:
	# starts the InFactory Worker
	poetry run python infactory_api/worker.py

.PHONY: backend
backend:
	# start the InFactory backend
	./start.sh

.PHONY: cleanEnv
cleanEnv:
	# Clean the environment
	lsof -ti :8000 | xargs kill -9 & deactivate & python3 -m venv venv & source venv/bin/activate & pip install . & rm -rf node_modules/.prisma

.PHONY: dozzle-up
dozzle-up:
	docker run -d --name dozzle --network nf-network -p 8080:8080 -v /var/run/docker.sock:/var/run/docker.sock amir20/dozzle:latest

.PHONY: dozzle-down
dozzle-down:
	docker stop dozzle
	docker rm dozzle

format:	## Run code autoformatters (black).
	pre-commit install
	git ls-files | xargs pre-commit run black --files

lint:	## Run linters: pre-commit (black, ruff, codespell) and mypy
	pre-commit install && git ls-files | xargs pre-commit run --show-diff-on-failure --files

test:	## Run tests via pytest.
	pytest tests
