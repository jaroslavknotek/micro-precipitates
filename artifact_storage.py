import io
import json
from PIL import Image 
import os
from datetime import datetime
from azure.storage.blob import BlobServiceClient,BlobClient,ContentSettings

DEFAULT_CONTAINER_NAME = "precipitatessegmentation"
container_name = os.environ.get("SegmentationContainerName",DEFAULT_CONTAINER_NAME)
storage_connection_string = os.environ.get('segmentationstoragesonnectionstring')
assert storage_connection_string, "Connection string env var not set"


def upload_artifacts(img,pred,dataframe,meta):
 
    ts = datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    blob_service_client = BlobServiceClient.from_connection_string(
        storage_connection_string,
    )

    file_descriptors = [
        ('img.png',_img_to_bytes(img), 'image/png'),
        ('pred.png',_img_to_bytes(pred),'image/png'),
        ('data.csv',_df_to_bytes(dataframe),'text/csv'),
        ('metadata.json',_json_to_bytes(meta),'application/json')
    ]   
    for file_name,file_bytes,ct in file_descriptors:
        blob_name = f"{ts}/{file_name}"
        sas = upload_file(
            blob_service_client,
            container_name,
            blob_name,
            file_bytes,
            ct
        ) 

def upload_file(
    blob_service_client, 
    container_name,
    blob_name,
    file_bytes,
    content_type
):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, 
        blob=blob_name
    )
    cs = ContentSettings(content_type=content_type)

    blob_client.upload_blob(
        file_bytes,
        blob_type="BlockBlob",
        content_settings=cs
    )

def _img_to_bytes(img):
    in_mem_file = io.BytesIO()
    res = Image.fromarray(img).convert('L')
    res.save(in_mem_file, format='PNG')  
    in_mem_file.seek(0)
    return in_mem_file.read()

def _df_to_bytes(df):
    s_buf = io.StringIO()
    df.to_csv(s_buf)
    return s_buf.getvalue().encode('utf-8')


def _json_to_bytes(json_data):
    return json.dumps(json_data).encode('utf-8')
