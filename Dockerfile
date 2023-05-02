FROM hdgigante/python-opencv
COPY . /app
WORKDIR /app/

RUN python3 -m ensurepip
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 80

CMD python3 -m streamlit run --server.port 80 ui_streamlit.py

