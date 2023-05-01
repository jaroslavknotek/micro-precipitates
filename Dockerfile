FROM python:3.9
COPY . /app
WORKDIR /app/

RUN ./scripts/setup_container.sh
# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir .
EXPOSE 8080

CMD python -m streamlit run --server.port 8080 ui_streamlit.py

