# Weichao Qiu @ 2018
# Convert vdb dataset to coco.
import vdb
import imageio as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os, json, errno, shutil
from pprint import pprint
import threading
import _mask as mask
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

class AtomicCounter:
    """An atomic, thread-safe incrementing counter.
    from: https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7
    """
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        with self._lock:
            self.value += num
            return self.value

img_id_counter = AtomicCounter()
anno_id_counter = AtomicCounter()

def make_coco_image(coco, dataset, frame_id, cam_name):
    ''' Make a coco image, thread-safe and can be parallelized '''
    ids = [frame_id]
    cams = [cam_name]
    images = dataset.get_image(cams, ids)

    # Extract annotation for per object
    img = io.imread(images[0]) 
    # TODO: It is not neccessary to read the image, only header is needed
    # But it is neccessary to read the seg mask
    img_id = img_id_counter.increment()

    # Convert file_name to relative filename
    # file_name = os.path.relpath(images[0], coco['data_dir_abs'])
    ext = os.path.splitext(images[0])[1]
    file_name = '%08d%s' % (img_id, ext)
    # Copy image to the target folder
    # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python
    shutil.copy2(images[0], os.path.join(coco['data_dir_abs'], file_name))

    image = {
        "license": 1,
        "file_name": file_name,
        "coco_url": "",
        "height": img.shape[0],
        "width": img.shape[1],
        "date_captured": "0000-00-00 00:00:00",
        "flickr_url": "",
        "id": img_id
    }
    print('Processing image id: ', img_id)

    # List all objects in this frame
    annotation_color = dataset.get_annotation_color(ids)
    segmasks = dataset.get_seg(cams, ids)
    segmask = io.imread(segmasks[0])

    annotations = []
    for actor_name, color in annotation_color.items():
        binary_mask = vdb.get_obj_mask(segmask, color)
        area = int(binary_mask.sum())

        if area == 0: continue # Filter out invalid object

        y, x = np.where(binary_mask == True)
        bbox = [int(x.min()), int(y.min()), int(x.max() - x.min()), int(y.max() - y.min())]
        # bbox format [x,y,width,height]

        anno_id = anno_id_counter.increment()

        # Make segmentation 
        # convert binary mask to coco segmentation format
        # (must have type np.ndarray(dtype=uint8) in column-major order)
        bimask = np.asfortranarray(binary_mask).astype('uint8')
        h, w = bimask.shape
        # segmentation = mask.encode(bimask.reshape((h, w, 1), order='F'))
        segmentation = mask.encode_uncompressed(bimask.reshape((h, w, 1), order='F'))[0]
        # mask.frUncompressedRLE(segmentation, h, w) 
        # rleUncompressed = mask._frString(segmentation)
        # https://github.com/facebookresearch/Detectron/issues/100
        # How to convert segmentation to a list?
        # TODO: Use mask API to speedup bbox and area computation

        annotation = {
            "segmentation": segmentation,
            "area": area, 
            "iscrowd": 1, # 1 means RLE encoding
            "image_id": img_id,
            "bbox": bbox,
            "category_id": 1, # TODO: Add category info
            "id": anno_id
        }
        annotations.append(annotation)

    return image, annotations

def make_category(id, name, supercategory):
    category = {
        "id": id,
        "name": name,
        "supercategory": supercategory,
    }
    return category

def mkdirp(data_dir):
    # the actual code
    try:
        os.makedirs(data_dir)
    except OSError as exc: 
        print('Failed to makedirp ' + data_dir)
        if exc.errno == errno.EEXIST and os.path.isdir(data_dir):
            pass

def to_coco():
    ''' Support object mask and bounding box '''
    db_root_dir = '/mnt/c/qiuwch/workspace/research_scripts/OWIMap'
    dataset = vdb.Dataset(db_root_dir)

    db_name = os.path.basename(db_root_dir)
    coco_data_dir = './' + db_name 
    ann_filename = '{coco_data_dir}/annotations/instances_{db_name}.json'.format(**locals())
    mkdirp('{coco_data_dir}/annotations/'.format(**locals()))

    # Create instances file
    # Fill in the details as needed
    info = {
        "description": "",
        "url": "",
        "version": "",
        "year": 2018,
        "contributor": "",
        "date_created": "",
    }

    licenses = [
        {
            "url": "",
            "id": 1,
            "name": "",
        }
    ]

    categories = [
        make_category(1, "person", "person"),
        make_category(2, "vehicle", "car"),
    ]

    images = []
    annotations = []

    ids = dataset.get_ids() # The frame ids in the dataset
    cam_name = dataset.get_cams()[0]

    # coco db information
    coco = {
        "data_dir" : coco_data_dir,
        "data_dir_abs" : os.path.abspath(coco_data_dir),
    }

    # Iterate over all frame ids
    for id in ids:
        image, annotation = make_coco_image(coco, dataset, id, cam_name)
        images.append(image)
        annotations += annotation

    data = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    # Create annotation file
    with open(ann_filename, 'w') as f:
        json.dump(data, f, indent=4)

def show_coco():
    db_name = 'OWIMap'
    db_root = db_name
    annFile = os.path.join(db_root, 'annotations/instances_{db_name}.json'.format(**locals()))

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    # catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
    catIds = coco.getCatIds(catNms=nms)
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[0])[0]

    # load and display image
    # use url to load image
    print(img['coco_url'])
    img_filename = os.path.join(db_root, img['file_name'])
    I = io.imread(img_filename)

    # load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

if __name__ == '__main__':
    to_coco()
    show_coco()
