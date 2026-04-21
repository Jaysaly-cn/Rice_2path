import os
import time
import zipfile
import pandas as pd
from pygbif import occurrences as occ

def run_dwca_pipeline():
    # Credentials
    user = "zhuyu"
    pwd = "aa123456789!"
    email = "1720216422@qq.com"

    # Predicates
    queries = [
        "taxonKey = 7015",
        "hasCoordinate = true",
        "mediaType = StillImage"
    ]

    try:
        key_list = occ.download(queries, user=user, pwd=pwd, email=email)
        download_key = key_list[0]
        print(f"Download Key: {download_key}")
    except Exception as e:
        print(e)
        return

    while True:
        time.sleep(15)
        meta = occ.download_meta(download_key)
        status = meta['status']
        print(f"Current Status: {status}")

        if status == 'SUCCEEDED':
            break
        elif status in ['KILLED', 'CANCELLED', 'FAILED']:
            return

    occ.download_get(download_key, path='.')

    zip_file = f"{download_key}.zip"
    extract_path = f"dwca_{download_key}"
    
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_path)

    occ_file = os.path.join(extract_path, "occurrence.txt")
    media_file = os.path.join(extract_path, "multimedia.txt")

    df_occ = pd.read_csv(
        occ_file, 
        sep='\t', 
        usecols=['gbifID', 'species', 'decimalLatitude', 'decimalLongitude', 'countryCode'], 
        on_bad_lines='skip',
        low_memory=False
    )

    if os.path.exists(media_file):
        df_media = pd.read_csv(
            media_file, 
            sep='\t', 
            usecols=['gbifID', 'identifier', 'type'], 
            on_bad_lines='skip'
        )
        
        df_media = df_media[df_media['type'] == 'StillImage']
        
        df_final = pd.merge(df_occ, df_media, on='gbifID', how='inner')
        df_final.rename(columns={'identifier': 'image_url'}, inplace=True)
        
        df_final.to_csv("noctuidae_100k_final.csv", index=False)
        print(f"Saved {len(df_final)} records.")
    else:
        df_occ.to_csv("noctuidae_100k_occ_only.csv", index=False)

if __name__ == "__main__":
    run_dwca_pipeline()