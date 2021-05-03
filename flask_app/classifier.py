from keras.models import load_model
import cv2 as cv
import numpy as np
from pathlib import Path
import warnings
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

warnings.filterwarnings("ignore")
#warnings.filterwarnings('ignore', category=DeprecationWarning)
#warnings.filterwarnings('ignore', category=FutureWarning)


DENOISED_MODEL_PATH = 'models/denoised/'
CLASSIF_MODEL_PATH = 'models/model_id_0/'

denoised_model = load_model(DENOISED_MODEL_PATH)
print('Denoising Model loaded.')
classif_model = load_model(CLASSIF_MODEL_PATH)
print('Classification Model loaded.')

class_ids = {"10μm": 0, "20μm": 1, "MCF7": 2, "HepG2": 3, "RBC": 4, "WBC": 5}
# setup plot details
colors = ['royalblue', 'moccasin', 'darkorange',
          'yellow', 'aquamarine', 'chartreuse']
class_names = ["10μm", "20μm", "MCF7", "HepG2", "RBC", "WBC"]
color_rgb = [(65, 105, 225), (255, 228, 181), (255, 140, 0),
             (255, 255, 0), (127, 255, 212), (127, 255, 0)]

#####inputs:    return cell_id, cell_array


"""# Data Formatting"""
#reshape and normalize images


def test_eval(xtest):
    xtest = np.array(xtest).reshape(-1, 66, 66, 1).astype('float32')
    xtest = xtest/255.
    return xtest

#resize to input image shape


def testcell_resizer(xtest, new_size=50, orig_size=66):
    pad = int((orig_size-new_size)/2)
    starti = 0 + pad
    endi = 66 - pad

    xts = []
    for i in xtest:
        xts.append(i[starti:endi, starti:endi])
    xts = np.asarray(xts).astype('float32')

    return xts

#returns list of cell id and predicted labels


def code_runner(xtest, class_ids, cell_ids):
    #denoiser
    denoised_imgs = denoised_model.predict(xtest)
    #classifier
    y_pred = classif_model.predict_classes(denoised_imgs)

    return list(zip(cell_ids, y_pred))

#returns generated image graphs for cell counts and micrographs with detection results
# MAKE JSON FROM HERE---------------------------------------------------------------------------------------------


def create_result_image(path, test_val, img_dir, imgname):
    _, ypred = zip(*test_val)
    cell_counts = [ypred.count(i) for i in range(len(class_names))]

    img = cv.imread(path, 0)
    I = img.copy()
    I = cv.cvtColor(I, cv.COLOR_GRAY2RGB)

    for cell in test_val:
        cell_id, ypred = cell[0], cell[1]

        pos = cell_id.find('#')
        x, y = int(cell_id[:pos]), int(cell_id[pos+1:])
        #print(cell_id, x, y)

        stpt = (x-33, y-33)
        endpt = (x+33, y+33)
        cv.rectangle(I, stpt, endpt,  color_rgb[ypred], 2)
    I = I[:, :, ::-1]
    final_img_path = f'static/{str("result_"+imgname)}'
    cv.imwrite(str(final_img_path), I)
    return cell_counts, final_img_path


def analyzer(imgname, img_dir, imgpath, cell_ids, cell_array):
    print("Analyzing Cells...")
    cell_ids = np.asarray(cell_ids)
    cell_array = np.asarray(cell_array)
    print(cell_array.shape)
    xtest = test_eval(cell_array)
    xtest = testcell_resizer(xtest)
    eval_results = code_runner(xtest, class_ids, cell_ids)
    cell_counts, image = create_result_image(path=str(imgpath), test_val=eval_results, img_dir=img_dir, imgname=imgname)
    print("Analyzing Complete...")
    return eval_results, cell_counts, image