import os
import json
import cv2
import numpy as np
import argparse
import torch
import torchreid
from mtcnn import MTCNN
from deepface import DeepFace
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_pipeline(base_dir, frame_set_name, output_dir, output_face_dir):
    FRAMES_DIR = os.path.join(base_dir, frame_set_name, 'frames')
    DETECTIONS_JSON = os.path.join(base_dir, frame_set_name, 'content', 'detections.json')

    FACE_THRESH = 0.6
    BODY_THRESH = 0.9

    assert torch.cuda.is_available(), "CUDA가 감지되지 않았습니다."
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_face_dir, exist_ok=True)

    face_detector = MTCNN()
    face_model_name = 'Facenet512'

    body_model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    ).to(DEVICE).eval()

    tracker = DeepSort(max_age=3, n_init=3, max_iou_distance=0.3)

    kf = tracker.tracker.kf
    frame_interval = 5
    for i in range(4):
        kf._motion_mat[i, i+4] = frame_interval

    person_gallery = {}
    next_person_id = 1
    final_id = {}
    pid_to_canonical_tid = {}

    TEMPLATE_5PTS = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)

    def align_face(img, kpts, output_size=(112,112)):
        M, _ = cv2.estimateAffinePartial2D(kpts, TEMPLATE_5PTS, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, output_size, borderValue=0)

    def extract_face_emb(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rep = DeepFace.represent(
            rgb,
            model_name=face_model_name,
            enforce_detection=False,
            detector_backend='mtcnn'
        )
        if not rep:
            return None
        emb = np.array(rep[0]['embedding'], dtype=np.float32)
        return emb / np.linalg.norm(emb)

    def extract_body_emb(img):
        t = (torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE)) / 255.0
        with torch.no_grad():
            feat = body_model(t)
        emb = feat.squeeze(0).cpu().numpy()
        return emb / np.linalg.norm(emb)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def assign_person_id(emb, mode):
        nonlocal next_person_id
        pid = f"person_{next_person_id}"
        person_gallery[pid] = {'face': None, 'body': None}
        person_gallery[pid][mode] = emb
        next_person_id += 1
        return pid

    def match_face(emb, alpha=0.6):
        best_id, best_sim = 'unknown', 0.0
        for pid, embs in person_gallery.items():
            ref = embs.get('face')
            if ref is None: continue
            sim = cosine(emb, ref)
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_sim >= FACE_THRESH:
            old = person_gallery[best_id]['face']
            updated = (1-alpha)*old + alpha*emb
            person_gallery[best_id]['face'] = updated / np.linalg.norm(updated)
            return best_id, best_sim
        return assign_person_id(emb, 'face'), 0.0

    def match_body(emb, alpha=0.7):
        best_id, best_sim = 'unknown', 0.0
        for pid, embs in person_gallery.items():
            ref = embs.get('body')
            if ref is None: continue
            sim = cosine(emb, ref)
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_sim >= BODY_THRESH:
            old = person_gallery[best_id]['body']
            updated = (1-alpha)*old + alpha*emb
            person_gallery[best_id]['body'] = updated / np.linalg.norm(updated)
            return best_id, best_sim
        return assign_person_id(emb, 'body'), 0.0

    with open(DETECTIONS_JSON, 'r') as f:
        dets = json.load(f)

    print("Using device:", DEVICE)
    print('Starting pipeline with dt=', frame_interval)

    for idx, fname in enumerate(tqdm(sorted(os.listdir(FRAMES_DIR)), desc='Pipeline')):
        if not fname.lower().endswith('.jpg'):
            continue

        frame = cv2.imread(os.path.join(FRAMES_DIR, fname))
        if frame is None:
            continue
        raw_frame = frame.copy()

        boxes = dets.get(fname, [])
        dets_list = [([x1,y1,x2-x1,y2-y1], conf, 'person') for x1,y1,x2,y2,conf in boxes]

        tracks = tracker.update_tracks(dets_list, frame=frame)
        curr_ids = {t.track_id for t in tracks}

        for old in list(final_id):
            if old not in curr_ids:
                final_id.pop(old)

        for t in tracks:
            raw_tid = t.track_id
            l, t_top, r, b = t.to_ltrb()
            x1, y1, x2, y2 = map(int, (l, t_top, r, b))
            roi = raw_frame[y1:y2, x1:x2]

            pid = final_id.get(raw_tid, 'unknown')
            if raw_tid in final_id:
                continue

            faces = face_detector.detect_faces(roi)
            if faces:
                x_f,y_f,w_f,h_f = faces[0]['box']
                kpts_roi = np.array([faces[0]['keypoints'][k] for k in
                                     ['left_eye','right_eye','nose','mouth_left','mouth_right']],
                                    dtype=np.float32)
                raw_face = roi[y_f:y_f+h_f, x_f:x_f+w_f]
                kpts_raw = kpts_roi - np.array([x_f,y_f], dtype=np.float32)
                aligned = align_face(raw_face, kpts_raw)

                emb_f = extract_face_emb(aligned)
                if emb_f is not None:
                    pid, sim = match_face(emb_f)
                    final_id[raw_tid] = pid
                    pid_to_canonical_tid.setdefault(pid, raw_tid)
                    continue

            emb_b = extract_body_emb(roi)
            pid, sim = match_body(emb_b)
            final_id[raw_tid] = pid
            pid_to_canonical_tid.setdefault(pid, raw_tid)

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, frame)

    print('Pipeline completed.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Identification Pipeline")
    parser.add_argument("--base_dir", type=str, required=True, help="기본 프레임/검출 경로")
    parser.add_argument("--frame_set_name", type=str, required=True, help="프레임 세트 폴더명")
    parser.add_argument("--output_dir", type=str, default="/content/output_pipeline_DeepSORT")
    parser.add_argument("--output_face_dir", type=str, default="/content/output_image")
    args = parser.parse_args()

    run_pipeline(args.base_dir, args.frame_set_name, args.output_dir, args.output_face_dir)
