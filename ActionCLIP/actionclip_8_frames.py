import numpy as np
import torch
from dotmap import DotMap
from numpy.random import randint
from PIL import Image
import yaml
import os
from modules.Visual_Prompt import visual_prompt
import clip
from utils.Text_Prompt import *
from utils.Augmentation import get_augmentation
from torch import nn
from argparse import ArgumentParser
import cv2
from collections import deque
import random

text_aug = [f"The person is {{}}" ,f"The man is {{}}", f"The woman is {{}}", f"The human is {{}}", f"a photo of action {{}}" ,f"a picture of action {{}}",
                f"a video of action {{}}", f"Human action of {{}}"]

parser = ArgumentParser()

parser.add_argument(
    '--actions', type=str, action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--pretrained', type=str, help='Input pretrained text.', default='checkpoint/vit-b-16-8f.pt'
)
parser.add_argument(
    '--config', type=str, help='Input config text.', default='configs/hmdb51/hmdb_test.yaml'
)

parser.add_argument(
    '--yolo', type=str, help='Input yolo version.', default='yolov8m.pt'
)

parser.add_argument(
    '--threshold', type=float, help='People detection threshold.', default=0.3
)

parser.add_argument(
    '--num_frames', type=int, help='Number of consecutive frames.', default=8
)

parser.add_argument(
    '--video_in', type=str, help='The video file input.', required=True
)

parser.add_argument(
    '--video_out', type=str, help='The video file output.', required=True
)

parser.add_argument(
    '--filter_action_prob', type=float, help='The probability to filter specific actions', default=0.4
)

parser.add_argument('--tracking_file', type=str, help='The tracking file.', required=True)
args = parser.parse_args()


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

class VideoRecord(object):
    def __init__(self):
        pass

    @property
    def path(self):
        return 'tmp'

    @property
    def num_frames(self):
        return 8

def remove_module_prefix(state_dict):

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove prefix 'module.'
        new_state_dict[new_key] = value
    return new_state_dict

def sample_indices(record):
    offsets = list()
    ticks = [i * record.num_frames // 8
             for i in range(8 + 1)]

    for i in range(8):
        tick_len = ticks[i + 1] - ticks[i]
        tick = ticks[i]
        if tick_len >= 1:
            tick += randint(tick_len - 1 + 1)
        offsets.extend([j for j in range(tick, tick + 1)])
    return np.array(offsets) + 1

def transform_images(config):
    record = VideoRecord()
    segment_indices = sample_indices(record)
    return get_data(record, segment_indices, config)

def load_image(directory, idx):
    return [Image.open(os.path.join(directory, 'img_{:05d}.png'.format(idx))).convert('RGB')]

def get_data(record, indices, config):
    images = list()
    for i, seg_ind in enumerate(indices):
        p = int(seg_ind)
        try:
            seg_imgs = load_image(record.path, p)
        except OSError:
            print('ERROR: Could not read image "{}"'.format(record.path))
            print('invalid indices: {}'.format(indices))
            raise
        images.extend(seg_imgs)
    transform = get_augmentation(False, config)
    process_data = transform(images)
    return process_data

def text_prompt():
    # text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
    #             f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
    #             f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
    #             f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
    #             f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)
    data_classes = [[i, action] for i, action in enumerate(args.actions)]

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data_classes])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict

def predict_action(frames, id2action, model, fusion_model, text_features, config, device):
    tmp_folder = 'tmp'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    for i, frame in enumerate(frames):
        cv2.imwrite('tmp/img_{:05d}.png'.format(i + 1), frame)
    image = transform_images(config).unsqueeze(dim=0)
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
    # if prob < args.filter_action_prob:
    #     return 'others'
    if id2action[idx] not in ['walk', 'sit', 'stand', 'interact']:
        return 'others'
    return id2action[idx]

def main():
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    id2action = {k: v for k, v in enumerate(args.actions)}

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load('ViT-B/16', device=device,
                                       jit=False)  # Must set jit=False for training  ViT-B/32

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    classes, _, _ = text_prompt()

    if os.path.exists(os.path.join('checkpoints', args.pretrained)):
        checkpoint = torch.load(os.path.join('checkpoints', args.pretrained), map_location='cpu') if device == 'cpu' \
        else torch.load(os.path.join('checkpoints', args.pretrained))
        model.load_state_dict(remove_module_prefix(checkpoint['model_state_dict']))
        fusion_model.load_state_dict(remove_module_prefix(checkpoint['fusion_model_state_dict']))

    fusion_model = fusion_model.to(device)
    model.eval()
    fusion_model.eval()

    text_inputs = classes.to(device)
    text_features = model.encode_text(text_inputs)

    video_path = args.video_in
    video_final_path = os.path.join('video_in', video_path + '.mp4')
    people_detection_threshold = args.threshold
    video_out_path = os.path.join('video_out', args.video_out + '.mp4')

    cap = cv2.VideoCapture(video_final_path)
    ret, frame = cap.read()
    w, h = frame.shape[1], frame.shape[0]
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                              (w, h))

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    frame_cnt = 0
    object_frames = {}  # Dictionary to store frames for each object
    datas = {}
    with open(os.path.join('tracking_file', args.tracking_file), 'r') as f:
        for line in f:
            cur_frame, person_id, x1, y1, x2, y2 = line.strip().split(',')[1:7]
            cur_frame = int(float(cur_frame))
            if cur_frame not in datas:
                datas[cur_frame] = []
            datas[cur_frame].append([int(person_id), int(float(x1)), int(float(y1)), int(float(x2) + float(x1)), int(float(y2) + float(y1))])
    print("Loaded data")
    
    with open(f'result_txt/{video_path}_{people_detection_threshold}.txt', 'w') as file:
        while ret:
            annotated_frame = frame.copy()
            if frame_cnt in datas:
                for data in datas[frame_cnt]:
                    person_id, x11, y11, x22, y22 = data
                    x1 = max(int(x11 - 40), 0)
                    y1 = max(int(y11 - 40), 0)
                    x2 = min(int(x22 + 40), w)
                    y2 = min(int(y22 + 40), h)
                    x11 = max(int(x11), 0)
                    y11 = max(int(y11), 0)
                    x22 = max(int(x22), 0)
                    y22 = max(int(y22), 0)

                    if person_id not in object_frames:
                        object_frames[person_id] = deque(maxlen=args.num_frames)

                    object_frames[person_id].append(frame[y1:y2, x1:x2])
                    text = f'{frame_cnt},{person_id},{x11},{y11},{x22},{y22},'
                    action = "Detecting..."
                    if frame_cnt == args.num_frames - 1:
                        if len(object_frames[person_id]) == args.num_frames:
                            action = predict_action(object_frames[person_id], id2action, model, fusion_model, text_features, config, device)
                        else:
                            tmp_object_frames = list(object_frames[person_id]) + [frame[y1:y2, x1:x2]] * (args.num_frames - len(object_frames[person_id]))
                            action = predict_action(tmp_object_frames, id2action, model, fusion_model, text_features, config, device)
                    
                    text += action + '\n'

                    annotated_frame = cv2.rectangle(annotated_frame, (x11, y11), (x22, y22),
                                                    (colors[person_id % len(colors)]), 2)
                    annotated_frame = cv2.putText(annotated_frame, action, (x11 + 10, y11 + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9, (0, 0, 255), 2)
                    file.write(text)

            cap_out.write(annotated_frame)
            ret, frame = cap.read()
            frame_cnt += 1
            print(f'Finished {frame_cnt} frames')
        
        cap.release()
        cap_out.release()
        cv2.destroyAllWindows()

    print("DONE")

if __name__ == '__main__':
    main()


