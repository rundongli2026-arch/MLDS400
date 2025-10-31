FROM rocker/r-base:4.4.1

WORKDIR /app

COPY src/r_titanic/install_packages.R ./install_packages.R
RUN Rscript install_packages.R

COPY src/r_titanic ./src/

COPY src/data ./data

CMD ["Rscript", "src/titanic_model.R"]
