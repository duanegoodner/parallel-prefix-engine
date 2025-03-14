FROM debian:latest
ARG PRIMARY_UID
ARG PRIMARY_GID
ARG PRIMARY_USER
ARG COMPONENTS=./image_resources/components

SHELL [ "/bin/bash", "--login", "-c" ]

COPY ${COMPONENTS}/create_sudo_user /tmp/create_sudo_user
RUN /tmp/create_sudo_user/main.sh ${PRIMARY_USER} ${PRIMARY_UID} ${PRIMARY_GID}\
    && sudo rm -rf /tmp/create_sudo_user

USER ${PRIMARY_USER}
ARG BUILD_DIR=/home/${PRIMARY_USER}/docker_build
WORKDIR ${BUILD_DIR}

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libpq-dev \
    git \
    curl \
    wget \
    zsh \
    nano \
    && sudo rm -rf /var/lib/apt/lists/*

COPY ${COMPONENTS}/zsh_setup ${BUILD_DIR}/zsh_setup
RUN ${BUILD_DIR}/zsh_setup/main.sh ${PRIMARY_USER} \
    && sudo rm -rf ${BUILD_DIR}/*

RUN mkdir -p /home/${PRIMARY_USER}/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/${PRIMARY_USER}/miniconda3/miniconda.sh \
    && bash /home/${PRIMARY_USER}/miniconda3/miniconda.sh -b -u -p /home/${PRIMARY_USER}/miniconda3 \
    && rm /home/${PRIMARY_USER}/miniconda3/miniconda.sh \
    && source ~/miniconda3/bin/activate \
    && conda init --all

# Install Poetry (as PRIMARY_USER)
# RUN curl -sSL https://install.python-poetry.org | python3 -

# # Ensure Poetry is in PRIMARY_USER's PATH
# ENV PATH="/home/${PRIMARY_USER}/.local/bin:$PATH"

# # Persist Poetry's path in user's profile for interactive sessions
# RUN echo 'export PATH="/home/${PRIMARY_USER}/.local/bin:$PATH"' >> /home/${PRIMARY_USER}/.bashrc && \
#     echo 'export PATH="/home/${PRIMARY_USER}/.local/bin:$PATH"' >> /home/${PRIMARY_USER}/.profile

RUN sudo mkdir /home/${PRIMARY_USER}/project
WORKDIR /home/${PRIMARY_USER}/project
RUN sudo rm -rf ${BUILD_DIR}

RUN sudo mkdir -p /usr/local/entrypoints
COPY ./image_resources/entrypoint_simple_persistence.sh /usr/local/entrypoints
