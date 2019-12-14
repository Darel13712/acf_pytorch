import zipfile
import os
import sys
import requests


def download_url(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def download(dataset='ml-latest-small'):
    """
    Download dataset from https://grouplens.org/datasets/movielens/
    Available: ml-20m, ml-latest-small, ml-latest and other.
    """

    print(f'Downloading {dataset} from grouplens...')
    archive = dataset + '.zip'

    try:
        download_url(f'http://files.grouplens.org/datasets/movielens/{archive}', archive)
        # urllib.request.urlretrieve(f'http://files.grouplens.org/datasets/movielens/{archive}', archive)
        print('Extracting files...')
        zip_ref = zipfile.ZipFile(archive, 'r')
        zip_ref.extractall()

        if os.path.exists(archive):
            os.remove(archive)
        print('Done\n')
    except Exception as e:
        print(e)

