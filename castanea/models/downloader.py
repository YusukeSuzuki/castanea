from pathlib import Path
import scipy.io as sio

def download_mat_file(filepath, fileurl):
    filepath = Path(VGG16MAT_FILENAME)

    if filepath.exists():
        print('{} exists on local'.format(VGG16MAT_FILENAME), file=sys.stderr)
    else:
        print('{} not exists on local, download from {}'.format(
            VGG16MAT_FILENAME, VGG16MAT_URL), file=sys.stderr)

        with urllib.request.urlopen(VGG16MAT_URL) as response, open(VGG16MAT_FILENAME, 'wb') as f:
            shutil.copyfileobj(response, f)
    
    x = sio.loadmat(filepath)
    return x

