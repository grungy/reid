FROM postgres:15.7

# Switch to root for installation
USER root

# Install necessary build dependencies with GPG fix
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    wget && \
    # Fix GPG issues by updating keyring and adding missing keys
    apt-get install -y debian-archive-keyring && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 648ACFD622F3D138 && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 0E98404D386FA1D9 && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys DCC9EFBF77E11517 && \
    # Update PostgreSQL repository key
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    postgresql-server-dev-15 \
    build-essential \
    git \
    gosu

# Clone pgvector repository and build
WORKDIR /tmp
RUN echo "Cloning pgvector..." && \
    git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    echo "Building pgvector..." && \
    make && \
    echo "Installing pgvector..." && \
    make install

# Clean up to reduce image size
RUN cd /tmp && \
    rm -rf pgvector && \
    apt-get remove -y --purge git build-essential postgresql-server-dev-15 wget gnupg2 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a custom entrypoint wrapper to ensure vector tables are set up on every startup
COPY setup-vector-tables.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/setup-vector-tables.sh

# Create a wrapper script that calls both the original entrypoint and our custom script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Store the command line arguments to pass to the original entrypoint\n\
ARGS=("$@")\n\
\n\
# First run the original PostgreSQL entrypoint script in the background\n\
/usr/local/bin/docker-entrypoint.sh "${ARGS[@]}" &\n\
PG_PROCESS=$!\n\
\n\
# Wait for PostgreSQL to start\n\
echo "Waiting for PostgreSQL to start..."\n\
until pg_isready -U postgres; do\n\
  echo "Still waiting for PostgreSQL to start..."\n\
  sleep 2\n\
done\n\
echo "PostgreSQL is up - executing vector setup script"\n\
\n\
# Run the setup script but don\'t fail if it has errors\n\
# This ensures the container stays running even if the vector setup fails\n\
if [ "$(whoami)" = "postgres" ]; then\n\
  # If already running as postgres, run script directly\n\
  /usr/local/bin/setup-vector-tables.sh || {\n\
    echo "Warning: Vector setup script exited with error code $?, but container will continue running"\n\
  }\n\
else\n\
  # Otherwise switch to postgres user without using password\n\
  gosu postgres /usr/local/bin/setup-vector-tables.sh || {\n\
    echo "Warning: Vector setup script exited with error code $?, but container will continue running"\n\
  }\n\
fi\n\
\n\
echo "Vector setup completed, database will continue running"\n\
\n\
# Wait for the PostgreSQL process to finish\n\
wait $PG_PROCESS\n\
' > /usr/local/bin/custom-entrypoint.sh

RUN chmod +x /usr/local/bin/custom-entrypoint.sh

# Switch back to postgres user for security
USER postgres

# Use our custom entrypoint instead of the default PostgreSQL entrypoint
ENTRYPOINT ["/usr/local/bin/custom-entrypoint.sh"]
CMD ["postgres"]
