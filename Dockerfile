FROM huggingface/transformers-pytorch-gpu:4.35.2

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt
RUN MAX_JOBS=4 pip install flash-attn==2.5.9.post1 --no-build-isolation
RUN git clone https://github.com/philschmid/FastChat.git
RUN pip install -e "./FastChat[model_worker,llm_judge]"
RUN pip install matplotlib==3.7.3 tabulate==0.9.0

ENV DAGSTER_HOME /app/dagster_data
RUN mkdir -p $DAGSTER_HOME
ENV PYTHONPATH /app
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY rlhf_training rlhf_training 
CMD dagster dev -f rlhf_training/llm_rlhf.py -p 3000 -h 0.0.0.0
