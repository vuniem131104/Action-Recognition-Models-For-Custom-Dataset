import torch
from PIL import Image
import os
from modules.Visual_Prompt import visual_prompt
import clip
from utils.Text_Prompt import *
from datasets.transforms_ss import *
import torchvision
from torch import nn
from argparse import ArgumentParser
import cv2
from collections import deque
from pymilvus import MilvusClient
import json
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") 
        new_state_dict[new_key] = value
    return new_state_dict

def text_prompt(text_aug, actions):
    text_dict = {}
    num_text_aug = len(text_aug)
    data_classes = [[i, action] for i, action in enumerate(actions)]

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data_classes])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
def get_augmentation():
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = data['ActionCLIP_args']['scale_size']

    unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                 GroupCenterCrop(data['ActionCLIP_args']['image_size'])])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])

def transform_images():
    segment_indices = [1,2,3,4,5,6,7,8]
    images = list()
    for seg_ind in segment_indices:
        try:
            seg_imgs = load_image(data['ActionCLIP_args']['images_folder'], seg_ind)
        except OSError:
            print('ERROR: Could not read image "{}"'.format(data['ActionCLIP_args']['images_folder']))
            print('invalid indices: {}'.format(segment_indices))
            raise
        images.extend(seg_imgs)
    transform = get_augmentation()
    process_data = transform(images)
    return process_data

def load_image(directory, idx):
    return [Image.open(os.path.join(directory, 'img_{:05d}.png'.format(idx))).convert('RGB')]

def predict_action(frames, id2action, model, fusion_model, text_features, device, text_aug):
    tmp_folder = data['ActionCLIP_args']['images_folder']
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    for i, frame in enumerate(frames):
        cv2.imwrite('{}/img_{:05d}.png'.format(data['ActionCLIP_args']['images_folder'], i + 1), frame)
    image = transform_images().unsqueeze(dim=0)
    with torch.no_grad():
        image = image.view((-1, 8, 3) + image.size()[-2:])
        b, t, c, h, w = image.size()
        image_input = image.to(device).view(-1, c, h, w)
        image_features = model.encode_image(image_input).view(b, t, -1)
        image_features = fusion_model(image_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        similarity = similarity.view(b, len(text_aug), -1).softmax(dim=-1)
        similarity = similarity.mean(dim=1, keepdim=False)
        idx = similarity.argmax(dim=1).item()
        prob = torch.max(similarity, dim=1)[0]
        
    return id2action[idx] if prob > args.filter_action_prob else 'doing others'

def predict_action_video(video_id, query_list, id2action, model, fusion_model, text_features, text_aug):
    results = list()
    video_path = os.path.join(video_id + '.mp4')

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    object_frames = {} 
    datas = {}
    
    for query_item in query_list:
        cur_frame, person_id, x1, y1, x2, y2, _ = query_item[1:]
        if cur_frame not in datas:
            datas[cur_frame] = []
        datas[cur_frame].append([int(person_id), x1, y1, x2, y2])
            
    first_item = next(iter(datas.items()))
    frame_cnt = (first_item[0] // 8) * 8
    frame_id_predict = frame_cnt + 8
    
    while ret:
        if frame_cnt in datas:
            for data in datas[frame_cnt]:
                person_id, x11, y11, x22, y22 = data
                x1 = max(int(x11 - 40), 0)
                y1 = max(int(y11 - 40), 0)
                x2 = min(int(x22 + 40), w)
                y2 = min(int(y22 + 40), h)

                if person_id not in object_frames:
                    object_frames[person_id] = deque(maxlen=8)

                object_frames[person_id].append(frame[y1:y2, x1:x2])
    
                if frame_cnt == frame_id_predict - 1:
                    if len(object_frames[person_id]) == 8:
                        action = predict_action(object_frames[person_id], id2action, model, fusion_model, text_features, device, text_aug)
                    else:
                        tmp_object_frames = list(object_frames[person_id]) + [frame[y1:y2, x1:x2]] * (8 - len(object_frames[person_id]))
                        # out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (224,224))
                        # for frame_ in tmp_object_frames + tmp_object_frames:
                        #     out.write(cv2.resize(frame_, (224, 224)))
                        # out.release()
                        action = predict_action(tmp_object_frames, id2action, model, fusion_model, text_features, device, text_aug)

                    results.append([video_id, frame_cnt, person_id, x11, y11, x22, y22, action])
                

        ret, frame = cap.read()
        frame_cnt += 1
    
    cap.release()
    cv2.destroyAllWindows()

    return results

def push_to_milvus(action_results, db_name='./milvus.db', is_insert=False):
    
    dim = 2048
    num_people = len(action_results)
    features = [np.random.random((dim, )) for _ in range(num_people)]
    
    client = MilvusClient(db_name)   
     
    if not client.has_collection(collection_name=data['VectorDB_args']['collection_name']):
        client.create_collection(
            collection_name=data['VectorDB_args']['collection_name'],
            dimension=dim
        )
        
    res = client.query(
        collection_name=data['VectorDB_args']['collection_name'],
        output_fields=["count(*)"]
    )

    count = res[0]['count(*)']
    print(count)

    data_to_push = [{"id": i + count, "video_id": result[0], "frame_id": result[1], "person_id": result[2], "x1": result[3], "y1": result[4],
             "x2": result[5], "y2": result[6], "vector": feature, "action": result[7]} 
            for i, (result, feature)in enumerate(zip(action_results, features))]
    
    if is_insert:
        res = client.insert(
            collection_name=data['VectorDB_args']['collection_name'],
            data=data_to_push
        )
    
    # res = client.query(
    #     collection_name=data['VectorDB_args']['collection_name'],
    #     filter="action == 'using smartphone'",
    # )
    
    # res = client.search(
    #     collection_name=data['VectorDB_args']['collection_name'],
    #     data=[features[0]],
    #     limit=5
    # )
    
    print('Pushed to Milvus')
    
def crop_and_save_video(x1, y1, x2, y2, frame_id, video_id, out):
    video_path = f'{video_id}.mp4'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 7)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    cnt = frame_id - 7
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cropped_frame = frame[y1:y2, x1:x2]
        out.write(cv2.resize(cropped_frame, (224, 224)))
        cnt += 1
        if cnt == frame_id + 1:
            break
    
    cap.release()
    out.release()

def clustering(video_id, folder="clusters", db_file='./milvus.db', collection="collection", method='kmeans', n_cluster=5):
    if not os.path.exists(folder):
        os.makedirs(folder)

    client = MilvusClient(db_file)
    res = client.query(
            collection_name=collection,
            filter="action != ''",
            output_fields=['id', 'video_id', 'frame_id', 'person_id', 'x1', 'y1', 'x2', 'y2', 'vector', 'action']
        )

    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_cluster)

    data = []
    for item in res:
        data.append(np.array(item['vector'], dtype=np.float32))
        
    kmeans.fit(data)

    predicted_labels = kmeans.predict(data)

    for i in tqdm(range(len(predicted_labels))):
        if not os.path.exists(f'{folder}/cluster_{predicted_labels[i]}'):
            os.makedirs(f'{folder}/cluster_{predicted_labels[i]}')
        
        x1, y1, x2, y2, frame_id, action = res[i]['x1'], res[i]['y1'], res[i]['x2'], res[i]['y2'], res[i]['frame_id'], res[i]['action']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{folder}/cluster_{predicted_labels[i]}/video_{i}.mp4', fourcc, 15, (224, 224))
        crop_and_save_video(x1, y1, x2, y2, frame_id, video_id, out)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--video_id', type=str, help='Input video id.', required=True
    )

    parser.add_argument(
        '--config', type=str, help='Input config file.', required=True
    )

    parser.add_argument(
        '--filter_action_prob', type=float, help='The probability to filter specific actions', default=0.3
    )

    args = parser.parse_args()

    with open(args.config) as f:
        data = json.load(f)


    text_aug = [f"The person is {{}}" ,f"The man is {{}}", f"The woman is {{}}", f"The human is {{}}", f"a photo of action {{}}" ,f"a picture of action {{}}",
                f"a video of action {{}}", f"Human action of {{}}"]

    actions = data['ActionCLIP_args']['actions']

    id2action = {k: v for k, v in enumerate(actions)}

    device = "cuda" if torch.cuda.is_available() else "cpu"  

    model, clip_state_dict = clip.load(data['ActionCLIP_args']['model'], device=device,
                                        jit=False) 

    fusion_model = visual_prompt(data['ActionCLIP_args']['sim_header'], clip_state_dict, 8)

    classes, _, _ = text_prompt(text_aug, actions)

    if os.path.exists(data['ActionCLIP_args']['checkpoint']):
        checkpoint = torch.load(data['ActionCLIP_args']['checkpoint'], map_location='cpu') if device == 'cpu' \
        else torch.load(os.path.join('checkpoints', 'vit-b-16-8f.pt'))
        model.load_state_dict(remove_module_prefix(checkpoint['model_state_dict']))
        fusion_model.load_state_dict(remove_module_prefix(checkpoint['fusion_model_state_dict']))


    fusion_model = fusion_model.to(device)
    model.eval()
    fusion_model.eval()

    text_inputs = classes.to(device)
    text_features = model.encode_text(text_inputs)
    video_id = args.video_id
    query_list = [
            ["VTP_BDV2_1307_1280_720_180", 9, 1, 393, 212, 393+143, 212+301, 0.8740782737731934],
            ["VTP_BDV2_1307_1280_720_180", 9, 2, 730, 486, 730+178, 486+229, 0.8723974227905273],
            ["VTP_BDV2_1307_1280_720_180", 10, 1, 394, 211, 394+142, 211+300, 0.8733908534049988],
            ["VTP_BDV2_1307_1280_720_180", 10, 2, 730, 486, 730+178, 486+229, 0.8693277835845947],
            ["VTP_BDV2_1307_1280_720_180", 11, 1, 395, 210, 395+141, 210+300, 0.8766065835952759],
            ["VTP_BDV2_1307_1280_720_180", 11, 2, 730, 486, 730+178, 486+229, 0.8707200288772583],
            ["VTP_BDV2_1307_1280_720_180", 12, 1, 397, 209, 397+139, 209+300, 0.8821026086807251],
            ["VTP_BDV2_1307_1280_720_180", 12, 2, 729, 485, 729+179, 485+230, 0.8637683987617493],
            ["VTP_BDV2_1307_1280_720_180", 13, 1, 398, 208, 398+138, 208+300, 0.8834617733955383],
            ["VTP_BDV2_1307_1280_720_180", 13, 2, 729, 485, 729+179, 485+230, 0.8579920530319214],
            ["VTP_BDV2_1307_1280_720_180", 14, 1, 399, 207, 399+136, 207+299, 0.8839132189750671],
            ["VTP_BDV2_1307_1280_720_180", 14, 2, 729, 484, 729+179, 484+231, 0.8685857057571411],
            ["VTP_BDV2_1307_1280_720_180", 15, 1, 401, 207, 401+135, 207+299, 0.8842822909355164],
            ["VTP_BDV2_1307_1280_720_180", 15, 2, 729, 484, 729+179, 484+231, 0.8580175638198853]
        ]
    action_results = predict_action_video(video_id, query_list, id2action, model, fusion_model, text_features, text_aug)
    print(action_results)
    push_to_milvus(action_results, is_insert=True)
    
    # clustering('VTP_BDV2_1307_1280_720_20_1729_1749', data['Cluster_args']['cluster_folder'], 
    #            data['VectorDB_args']['database_file'], data['VectorDB_args']['collection_name'], 
    #            data['Cluster_args']['cluster_method'], data['Cluster_args']['n_clusters'])
    
    
