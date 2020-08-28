import requests
import json
import cv2
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

model_url = 'http://95.216.214.227:8501/v1/models/model:predict'


brands = [
    "Johnnie Walker",
    "Lenor",
    "Fa",
    "Три Медведя",
    "Amstel",
    "Пава-пава",
    "Taft",
    "Heineken",
    "Royal Canine",
    "Pantene",
    "Pedigree",
    "Дымов",
    "Jack Daniels",
    "Останкино",
    "Я",
    "Whiskas",
    "Tuborg",
    "Alpen Gold",
    "Kit-Kat",
    "Охота",
    "48 копеек",
    "Pepsi",
    "Elseve",
    "Mr.Proper",
    "Old Spice",
    "Черкизово",
    "Zhatetsky Gus",
    "Chivas Regal",
    "Nescafe",
    "Venus",
    "Балтика",
    "Слобода",
    "Nesquik",
    "Gillette",
    "Nutella",
    "Pampers",
    "Fitness",
    "Snickers",
    "Oral-B",
    "Heinz",
    "Tide",
    "Россия",
    "M&Ms",
    "Ballantines",
    "Nivea",
    "Head&Shoulders",
    "Felix",
    "Campo Viejo",
    "Петелинка",
    "Фруктовый Сад",
    "J7",
    "Ariel",
    "Lays",
    "Movenpick",
    "Shauma",
    "Арарат",
    "Fairy",
    "Always"]


def crop(img):
    img_center = [img.shape[0] // 2, img.shape[1] // 2]

    if img.shape[0] < img.shape[1]:
        img = img[:, img_center[1] - img_center[0]:img_center[1] + img_center[0]]
    else:
        img = img[img_center[0] - img_center[1]:img_center[0] + img_center[1], :]
    return img


def predict(img):
    
    image = np.array([img])
    
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(model_url, data=data, headers=headers)

    return json.loads(json_response.text)['predictions'][0]


def show_attention(image, mask):
    _mask = np.zeros((224, 224, 1), dtype='float32')
    _mask[:, :, 0] = cv2.resize(mask, (224, 224))
    _mask = _mask / np.max(_mask)
    _image = cv2.cvtColor((image * _mask * 255).astype('uint8'), cv2.COLOR_BGR2RGB)

    return _image


def segmentation_report(img, to_save=False):
    
    prediction = predict(img)
    masks = np.array(prediction['output_mask'])
    masks = masks > 0.95

    cm = get_colormap(masks.shape[-1])

    masks = [masks[:, :, i] for i in range(masks.shape[-1])]

    masks = list(map(lambda x: (x).astype('float32'), masks))
    masks = list(
        map(lambda x: (255 * cv2.resize(x, (224, 224), interpolation=cv2.INTER_NEAREST)).astype('uint8'), masks))
    masks = list(map(lambda x: cv2.medianBlur(x, 11).astype('bool'), masks))

    segmap = img.copy()
    observed = []

    for i in range(len(masks)):
        if np.sum(masks[i]) > 0:
            observed.append(i)

            _img = segmap.copy()

            _img[:, :, 0][masks[i]] = cm[i][0]
            _img[:, :, 1][masks[i]] = cm[i][1]
            _img[:, :, 2][masks[i]] = cm[i][2]

            opacity = 0.7
            cv2.addWeighted(_img, opacity, segmap, 1 - opacity, 0, segmap)

    if observed:
        patches = [mpatches.Patch(color=(cm[i][0], cm[i][1], cm[i][2]), label=brands[i]) for i in observed]
    else:
        patches = [mpatches.Patch(color=(1, 1, 1), label="Empty")]

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.imshow(segmap)

    if to_save:
        plt.savefig('output.png')


def get_colormap(num):
    cmap = matplotlib.cm.get_cmap('jet')
    return [cmap(i) for i in np.arange(0,1,1/num)]