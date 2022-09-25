FROM python:3.9.7
RUN mkdir /code
RUN apt update
RUN apt install -y gettext
RUN python -m pip install --upgrade pip

RUN pip install numpy
RUN pip install pandas
RUN pip install requests
RUN pip install networkx
RUN pip install gensim
RUN pip install spacy
RUN pip install nltk

COPY . /code
WORKDIR /code/
ENV OUTPUT_FOLDER='output_folder'
ENV INPUT_FOLDER='intput_folder'
CMD python baseline.py --input_folder ${INPUT_FOLDER} --output_folder ${OUTPUT_FOLDER}
