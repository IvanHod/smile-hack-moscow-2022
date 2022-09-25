FROM python:3.9.7
RUN mkdir /code
COPY requirements.txt /code
RUN pip install -r /code/requirements.txt
RUN apt update
RUN apt install -y gettext
COPY . /code
WORKDIR /code/
ENV OUTPUT_FOLDER='output_folder'
ENV INPUT_FOLDER='intput_folder'
CMD python baseline.py --input_folder ${INPUT_FOLDER} --output_folder ${OUTPUT_FOLDER}
