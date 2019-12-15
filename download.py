from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os
import time
import sys

# flickerのkey,secret入力
key = "xxxxxxxxxx"
secret = "xxxxxxxxxx"
wait_time = 1

keyword = sys.argv[1]
savedir = "./" + keyword

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text=keyword,
    per_page=400,
    media='photos',
    sort='relevance',
    safe_search=1,
    extras='url_q,license'
)

photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'

    # すでにダウンロード済みの画像はスキップ
    if os.path.exists(filepath):
        continue

    # ネット上からファイルをダウンロードし保存する
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
