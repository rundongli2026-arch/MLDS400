FROM rocker/r-base:4.4.1

WORKDIR /app
COPY src/r_titanic/install_packages.R ./install_packages.R
RUN Rscript install_packages.R

COPY src/ ./src/
CMD ["Rscript", "src/r_titanic/titanic_model.R"]
