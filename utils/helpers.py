#%%
import os
import tarfile
import cv2
import uuid
# Get base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Setup paths for data
def setup_paths():
    print("Setting up paths...")
    POSITIVE_PATH = os.path.join(BASE_DIR, 'data', 'positive')
    NEGATIVE_PATH = os.path.join(BASE_DIR, 'data', 'negative')
    TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train')
    os.makedirs(POSITIVE_PATH, exist_ok=True)
    os.makedirs(NEGATIVE_PATH, exist_ok=True)
    os.makedirs(TRAIN_PATH, exist_ok=True)
    print('Paths setup')

# Uncompress lfw dataset
def uncompress_lfwa():
    tar_path = os.path.join(BASE_DIR, 'data', 'lfw.tgz')
    extract_path = os.path.join(BASE_DIR, 'data', 'lfw')
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Tar path: {tar_path}")
    print(f"Checking existence of {tar_path}...")
    
    if not os.path.exists(tar_path):
        print(f"Tar file {tar_path} does not exist.")
        return
    
    print(f"Checking existence of {extract_path}...")
    if os.path.exists(extract_path):
        print(f"Extract path {extract_path} already exists.")
        return
    
    print("Uncompressing lfw dataset...")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(path=os.path.join(BASE_DIR, 'data'))
        print('Uncompressed lfw dataset')
    except tarfile.TarError as e:
        print(f"Error: {e}")

# Transfer lfw into data/negative
def transfer_lfw():
    lfw_path = os.path.join(BASE_DIR, 'data', 'lfw')
    negative_path = os.path.join(BASE_DIR, 'data', 'negative')
    
    if not os.path.exists(lfw_path):
        print(f"Path {lfw_path} does not exist.")
        return
    
    if not os.path.exists(negative_path):
        print(f"Path {negative_path} does not exist.")
        return
    
    for root, dirs, files in os.walk(lfw_path):
        for file in files:
            if file.endswith('.jpg'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(negative_path, file)
                os.rename(src_path, dst_path)
    print('Transfer complete')

# Acquiring positive and training data
def acquire_data():
    capture = cv2.VideoCapture(1)
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()
        # resize frame to 250x250
        frame=frame[120:120+250,200:200+250,:]
        
        # collect positive samples
        if cv2.waitKey(1) & 0xFF == ord('p'):
            image_name=os.path.join(BASE_DIR, 'data', 'positive', '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(image_name, frame)
        
        # collet training samples
        if cv2.waitKey(1) & 0xFF == ord('t'):
            image_name=os.path.join(BASE_DIR, 'data', 'train', '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(image_name, frame)
        
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
# %%
