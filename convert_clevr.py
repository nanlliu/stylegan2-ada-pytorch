import json
import numpy as np
from PIL import Image


if __name__ == '__main__':
    
    data = np.load('/root/relational_data_multi_objects_50000_train.npz')
    images = data['ims']
    labels = data['labels']
    
    labels_json = {'labels': []}
    
    for i in range(images.shape[0]):
        im = Image.fromarray(images[i])
        label = labels[i].tolist()
        save_path = '/root/clevr_dataset/CLEVR_new_{:06}.png'.format(i)
        im.save(save_path)
        labels_json['labels'].append(['CLEVR_new_{:06}.png'.format(i), label])
        print(save_path)

    json_path = '/root/clevr_dataset/dataset.json'
    with open(json_path, 'w') as f:
        json.dump(labels_json, f)

    print('save', json_path)
