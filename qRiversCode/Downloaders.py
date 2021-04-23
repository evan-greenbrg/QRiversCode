import os
from google.cloud import storage


def pullRiverFiles(outroot, bucket_name, river, year, 
                   stage=None, storage_client=storage.Client()):
    """
    Pulls all files associated with a river and year in a 
    given storage bucket
    """
    # Get prefix to search for files
    if stage:
        prefix = f'{river}/{stage}/{year}'
    else:
        prefix = f'{river}/{year}'

    # Set up bucket object
    bucket = storage_client.get_bucket(bucket_name)

    # Get objects in bucket with prefix
    blobs = bucket.list_blobs(prefix=prefix)

    # Iterate over file objects
    for blob in blobs:
        # split prefix to list
        namel = blob.name.split('/')
        fn = namel.pop(-1)

        # Join prefix path into local file path
        outpath = os.path.join(outroot, '/'.join(namel))

        # Check if path exists
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        outpath = os.path.join(outpath, fn)

        # download the object
        blob.download_to_filename(outpath)

        if os.path.exists(outpath):
            print('File finished downloading')
        else:
            print('File didn\'t download, something happened')


def pushRiverFiles(fn, bucket_name, river, stage, year, idx,
                   storage_client=storage.Client()):
    """
    Push file to an object in google cloud storage
    """
    bucket_name = 'earth-engine-rivmap'


    name = fn.split('/')[-1]

    path = f'{river}/{stage}/{year}/{idx}/{name}'

    bucket = storage_client.get_bucket(bucket_name)
    bl = bucket.blob(path)

    bl.upload_from_filename(fn)
    
    print('Uploaded: ', name)

    


if __name__ == '__main__':
#    outroot = '/Volumes/EGG-HD/PhD Documents/Projects/BarT/riverData'
    outroot = '/home/greenberg/ExtraSpace/PhD/Projects/BarT/'
    bucket_name = 'earth-engine-rivmap'
    river = 'beni'
    stage = 'clipped'
    year = 1985

    pullRiverFiles(outroot, bucket_name, river, stage, year)
