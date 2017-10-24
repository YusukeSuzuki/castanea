from pathlib import Path
import scipy.io as sio
import sys

def download_mat_file(filepath, fileurl):
    if Path(filepath).exists():
        print('{} exists on local'.format(filepath), file=sys.stderr)
    else:
        print('{} not exists on local, download from {}'.format(
            filepath, fileurl), file=sys.stderr)

        with urllib.request.urlopen(fileurl) as response, open(filepath, 'wb') as f:
            shutil.copyfileobj(response, f)
    
    x = sio.loadmat(filepath)
    return x

