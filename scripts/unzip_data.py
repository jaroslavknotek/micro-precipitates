import shutil
import pathlib
from zipfile import ZipFile
import sys
import logging
import os
import imageio

logging.basicConfig(level = logging.INFO)
def _map_file_to_folder(filenames):

    filename_mapping = {}
    for f in filenames:
        rev=f[::-1]
        rest= rev[rev.index('_')-1::-1]
        stem = rev[rev.index('_')+1:][::-1]

        key = rest.replace('original','img')
        filename_mapping.setdefault(stem,{})[key] = f

    return filename_mapping

def extract_archive(zip_path,output_dir):
    tmp_path = pathlib.Path('/tmp')
    with ZipFile(zip_path) as zip_archive:
        filenames = [
            child.filename 
            for child in zip_archive.infolist()
        ]

        mapping = _map_file_to_folder(filenames)
        for k,v in mapping.items():
            parts =  ["mask.png","img.png"]
            err = False
            for part in parts:
                if part not in v:
                    logging.warning(f"Part {part} not found in {k}")
                    err = True
            if err:
                # skipping because of missing part
                continue

            for part in parts:    
                destination = output_dir/k/part
                destination.parent.mkdir(
                    exist_ok=True,
                    parents= True
                )
                logging.info(
                    f"Extracting {v[part]} to {destination}"
                )
                
                tmp_des = tmp_path/v[part]
                try:
                    os.remove(tmp_des)
                except FileNotFoundError:
                    pass

                zip_archive.extract(
                    v[part],
                    path=tmp_path
                )
                if part == 'img.png':
                    shutil.move(tmp_des,destination)
                else:
                    mask_rgb = imageio.imread(tmp_des)
                    mask = mask_rgb[:,:,0]
                    imageio.imwrite(destination,mask)

if __name__ == "__main__":
    zip_path = pathlib.Path(sys.argv[1])
    output_dir = pathlib.Path(sys.argv[2])
    extract_archive(zip_path,output_dir)

