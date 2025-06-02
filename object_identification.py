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

def run_pipeline(base_dir, frame_set_name, output_dir, output_face_dir, output_video):
    """
    - JSON에서 사람(person) vs 비사람 객체 분류
    - person 객체는 MTCNN으로 얼굴 검출 + FaceNet/OSNet 임베딩
    - 비사람 객체는 위치(feature) 계산에만 사용
    - 얼굴이 없는 경우, 이전 프레임까지 등장한 후보(pids - 얼굴 매칭 완료된 pids)만 고려하여 PCB+LOC 매칭
      · OSNet 임베딩 raw cosine ≥ 0.8  → 위치 정보 무시, 오직 전신 임베딩으로 매칭
      · 0.7 ≤ raw cosine < 0.8       → score_pcb = raw cosine, 전신 가중치 0.6, 위치 가중치 0.4
      · raw cosine < 0.7             → score_pcb = 0, 전신 가중치 0.4, 위치 가중치 0.6
      · 위치 점수(score_loc)는 “diff → 1/(1+diff)” 형태로 [0,1] 구간에 매핑 후 공통 앵커 개수로 평균내어 [0,1] 유지
      · 거리 계산 시 “픽셀 거리 / sqrt(앵커 영역)” 형태로 정규화하여 prev_dist_norm, curr_dist_norm 계산
    - 사람별로 고유한 색으로 바운딩 박스 표시
    - 매 프레임마다 person_gallery 상태 요약 출력 (주석 처리)
    - 트래킹: 객체에 ID가 부여된 이후, 이미 할당된 트랙에 대해서는 분석 없이 시각화만 수행
    """

    # ── (1) 경로 설정 ───────────────────────────────────────────────────────────
    FRAMES_DIR      = os.path.join(base_dir, frame_set_name, 'frames')
    DETECTIONS_JSON = os.path.join(base_dir, frame_set_name, 'content', 'detections.json')

    # ── (2) 매칭 임계치, blending 비율 ─────────────────────────────────────────
    FACE_THRESH      = 0.6    # 얼굴 임베딩 유사도 임계치
    FACE_ALPHA       = 0.6    # 얼굴 임베딩 업데이트 blending 비율
    BODY_ALPHA       = 0.6    # 전신 임베딩 업데이트 blending 비율
    MATCH_THRESH     = 0.2    # PCB+위치 결합 매칭 임계치
    PCB_TRUST_THRESH = 0.7    # OSNet 임베딩 신뢰 임계치 (0.7)
    PCB_OVERRIDE     = 0.8    # OSNet raw cosine ≥ 0.8 → 위치 무시

    # ── (3) CUDA 설정 ─────────────────────────────────────────────────────────
    assert torch.cuda.is_available(), "CUDA가 필요합니다."
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # ── (4) 출력 디렉터리 생성 ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_face_dir, exist_ok=True)

    # ── (5) 얼굴/바디 모델 + 트래커 초기화 ──────────────────────────────────────
    face_detector = MTCNN()
    face_model_name = 'Facenet512'

    # OSNet-AIN 사용
    body_model = torchreid.models.build_model(
        name='pcb_p6',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    ).to(DEVICE).eval()

    tracker = DeepSort(
        max_age=3,
        n_init=3,
        max_iou_distance=0.1,
    )

    # ── (6) person_gallery, ID/색상 매핑 초기화 ─────────────────────────────────
    person_gallery = {}
    next_person_id = 1
    final_id = {}        # tracker_id -> person_id
    pid2color = {}       # person_id -> (B,G,R)

    def get_new_color():
        return tuple(np.random.randint(0, 256, size=3).tolist())

    # ── (7) 얼굴 정렬 템플릿 ─────────────────────────────────────────────────────
    TEMPLATE_5PTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def align_face(img, kpts):
        M, _ = cv2.estimateAffinePartial2D(kpts, TEMPLATE_5PTS, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (112,112), borderValue=0)

    def extract_face_emb(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rep = DeepFace.represent(
            rgb, model_name=face_model_name,
            enforce_detection=False, detector_backend='mtcnn'
        )
        if not rep:
            return None
        emb = np.asarray(rep[0]['embedding'], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def extract_body_emb(img):
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
        with torch.no_grad():
            feat = body_model(t).squeeze(0).cpu().numpy()
        return feat / (np.linalg.norm(feat) + 1e-6)

    def cosine(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6))

    def assign_person_id(emb, mode):
        nonlocal next_person_id
        pid = f"person_{next_person_id}"
        person_gallery[pid] = {'face': None, 'body': None, 'history': []}
        person_gallery[pid][mode] = emb
        pid2color[pid] = get_new_color()
        next_person_id += 1
        return pid

    def match_face(emb):
        best_id, best_sim = 'unknown', 0.0
        print("    [FACE MATCH] 비교 대상 person_ID별 얼굴 유사도:")
        for pid, embs in person_gallery.items():
            ref = embs.get('face')
            if ref is None:
                print(f"      {pid}: 얼굴 임베딩 없음")
                continue
            sim = cosine(emb, ref)
            print(f"      {pid}: cos_sim = {sim:.4f}")
            if sim > best_sim:
                best_id, best_sim = pid, sim

        if best_sim >= FACE_THRESH:
            old = person_gallery[best_id]['face']
            updated = (1 - FACE_ALPHA) * old + FACE_ALPHA * emb
            person_gallery[best_id]['face'] = updated / np.linalg.norm(updated)
            return best_id, best_sim

        return assign_person_id(emb, 'face'), 0.0

    def match_body_and_location(emb_curr, curr_anch_boxes, person_center, candidate_pids):
        """
        emb_curr: 현재 프레임에서 추출된 바디 임베딩
        curr_anch_boxes: 현재 프레임의 scene_anchors 사전,
                          {'chair': [bbox1, bbox2, ...], ...}
        person_center: 현재 프레임에서 사람 중심 좌표 (x, y)
        candidate_pids: 이전 프레임까지 등장했지만 아직 얼굴 매칭 안 된 pid 리스트
        """
        best_pid, best_score = None, -np.inf
        print("    [BODY+LOC MATCH] 후보 pids:", candidate_pids)

        for pid in candidate_pids:
            info = person_gallery[pid]
            prev_body = info['body']
            prev_history = info['history']

            # 1) PCB(OSNet) raw cosine 계산
            if prev_body is not None and emb_curr is not None:
                raw_score_pcb = cosine(prev_body, emb_curr)
                print(f"      PID={pid}: PCB(OSNet) raw_cosine = {raw_score_pcb:.4f}")
            else:
                raw_score_pcb = 0.0
                print(f"      PID={pid}: PCB(OSNet) 유사도 계산 불가 (face/body 임베딩 없음)")

            # 2) raw_score_pcb ≥ PCB_OVERRIDE → 위치 무시, 오직 raw_score_pcb로 매칭
            if raw_score_pcb >= PCB_OVERRIDE:
                score_combined = raw_score_pcb
                print(f"        → raw_score_pcb ({raw_score_pcb:.4f}) ≥ {PCB_OVERRIDE}, "
                      f"위치 정보 무시, combined = {score_combined:.4f}")
                if score_combined > best_score:
                    best_score, best_pid = score_combined, pid
                continue

            # 3) raw_score_pcb < PCB_OVERRIDE → 신뢰 임계치(0.7) 기반 score_pcb 결정
            if raw_score_pcb >= PCB_TRUST_THRESH:
                score_pcb = raw_score_pcb
                print(f"      PID={pid}: PCB(OSNet) raw ({raw_score_pcb:.4f}) ≥ {PCB_TRUST_THRESH} → used={score_pcb:.4f}")
            else:
                score_pcb = 0.0
                print(f"      PID={pid}: PCB(OSNet) raw ({raw_score_pcb:.4f}) < {PCB_TRUST_THRESH} → used=0.0000")

            # 4) location 유사도 계산 ([0,1] 범위로 정규화)
            score_loc = 0.0
            if prev_history:
                prev_anch = prev_history[-1]['anch_dist']  # {'chair': float_prev_dist_norm, ...}
                common_partials = []
                for cls, prev_dist_norm in prev_anch.items():
                    if cls not in curr_anch_boxes:
                        continue

                    # (a) 현재 프레임에서 cls 앵커 바운딩박스들로부터 정규화된 거리 리스트 계산
                    curr_dists_norm = []
                    for (bx1, by1, bx2, by2) in curr_anch_boxes[cls]:
                        c_x = (bx1 + bx2) // 2
                        c_y = (by1 + by2) // 2
                        d_raw = float(np.hypot(person_center[0] - c_x, person_center[1] - c_y))
                        area = float((bx2 - bx1) * (by2 - by1))
                        d_norm = d_raw / (np.sqrt(area) + 1e-6)
                        curr_dists_norm.append(d_norm)

                    if not curr_dists_norm:
                        continue

                    # (b) prev_dist_norm vs curr_dists_norm 간 최소 차이 사용
                    diffs = [abs(prev_dist_norm - d_curr) for d_curr in curr_dists_norm]
                    min_idx = int(np.argmin(diffs))
                    min_diff = diffs[min_idx]
                    chosen_curr_norm = curr_dists_norm[min_idx]

                    # partial_norm = 1 / (1 + diff) → [0,1]
                    partial_norm = 1.0 / (1.0 + min_diff)
                    common_partials.append(partial_norm)

                    print(
                        f"      PID={pid}: Anchor '{cls}' → prev_dist_norm={prev_dist_norm:.3f}, "
                        f"curr_best_norm={chosen_curr_norm:.3f}, diff={min_diff:.3f}, partial_norm={partial_norm:.3f}"
                    )

                if common_partials:
                    # 공통 앵커별 partial_norm의 평균 사용 (여전히 [0,1] 범위)
                    score_loc = float(np.mean(common_partials))
                else:
                    print(f"      PID={pid}: 공통 anchor 없음 → 위치 유사도=0")
                    score_loc = 0.0
            else:
                print(f"      PID={pid}: 이전 frame 히스토리 없음 → 위치 유사도=0")
                score_loc = 0.0

            # 5) 동적 가중치 할당
            if score_pcb > 0.0:
                # OSNet 임베딩 신뢰 시 전신 0.6, 위치 0.4
                w_pcb, w_loc = 0.6, 0.4
            else:
                # OSNet 임베딩 신뢰 불가 시 위치 0.6, 전신 0.4
                w_pcb, w_loc = 0.4, 0.6

            score_combined = w_pcb * score_pcb + w_loc * score_loc
            print(f"      PID={pid}: score_pcb={score_pcb:.4f}, score_loc={score_loc:.4f}, "
                  f"weights=(pcb:{w_pcb:.1f},loc:{w_loc:.1f}), combined={score_combined:.4f}")

            if score_combined > best_score:
                best_score, best_pid = score_combined, pid

        return best_pid, best_score

    # ── (8) JSON 로드 ────────────────────────────────────────────────────────────
    with open(DETECTIONS_JSON, 'r') as f:
        dets = json.load(f)

    print("Using device:", DEVICE)
    print("Starting pipeline…")

    # ── (9) 프레임별 처리 ───────────────────────────────────────────────────────
    for idx, fname in enumerate(tqdm(sorted(os.listdir(FRAMES_DIR)), desc='Pipeline')):
        if not fname.lower().endswith('.jpg'):
            continue

        frame = cv2.imread(os.path.join(FRAMES_DIR, fname))
        if frame is None:
            continue
        raw_frame = frame.copy()

        # (A) 비사람(anchor) 객체들만 모아서 scene_anchors 구성
        scene_anchors = {}
        for obj in dets.get(fname, []):
            cls = obj["class"]
            if cls != "person":
                scene_anchors.setdefault(cls, []).append(obj["bbox"])

        # (B) person 객체만 DeepSort 입력 리스트로 변환
        person_dets = [d for d in dets.get(fname, []) if d["class"] == "person"]
        dets_list = []
        for d in person_dets:
            x1, y1, x2, y2 = d["bbox"]
            conf           = d["conf"]
            dets_list.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        # ── (C) DeepSort 트래킹 업데이트 (먼저 트랙 ID 확보)  -----------------------
        tracks = tracker.update_tracks(dets_list, frame=frame)

        # ── (D) 얼굴 있는 객체와 없는 객체 분리 전처리 ───────────────────────────────
        # person_gallery 키 복사 → 이전 프레임까지 등장한 pid 리스트
        person_list = list(person_gallery.keys())

        face_tracks = []     # 얼굴이 검출된 트랙 정보 모아두기
        no_face_tracks = []  # 얼굴이 검출되지 않은 트랙 정보 모아두기

        for t in tracks:
            tid = t.track_id

            # (D.1) 이미 final_id에 매핑된 트랙은 바로 시각화만 수행
            if tid in final_id:
                pid = final_id[tid]
                color = pid2color[pid]
                l, t_top, r, b = t.to_ltrb()
                x1_t, y1_t, x2_t, y2_t = map(int, (l, t_top, r, b))
                cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), color, 2)
                cv2.putText(frame, f"{pid}", (x1_t, y1_t - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                #print(f"Frame {idx+1} | track_{tid} → {pid} (이미 ID 부여됨 → 스킵)")
                continue

            # (D.2) final_id에 없는 트랙 → ROI, 얼굴/바디 임베딩, 앵커 거리 정보 계산
            l, t_top, r, b = t.to_ltrb()
            x1, y1, x2, y2 = map(int, (l, t_top, r, b))
            roi = raw_frame[y1:y2, x1:x2]

            # 얼굴 임베딩 시도
            face_emb = None
            faces = face_detector.detect_faces(roi)
            if faces:
                x_f, y_f, w_f, h_f = faces[0]['box']
                kpts = np.array([faces[0]['keypoints'][k] for k in
                                ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']],
                                dtype=np.float32)
                raw_face = roi[y_f:y_f+h_f, x_f:x_f+w_f]
                aligned = align_face(raw_face, kpts - np.array([x_f, y_f], dtype=np.float32))
                emb_f = extract_face_emb(aligned)
                if emb_f is not None:
                    face_emb = emb_f

            # 바디 임베딩
            emb_b = extract_body_emb(roi)

            # 앵커 거리 정보(클래스당 앵커 bbox 리스트)
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            anch_dist_boxes = {}
            for cls, boxes in scene_anchors.items():
                anc_boxes = []
                for bx1, by1, bx2, by2 in boxes:
                    anc_boxes.append((bx1, by1, bx2, by2))
                anch_dist_boxes[cls] = anc_boxes

            # 얼굴 여부에 따라 리스트에 저장
            if face_emb is not None:
                face_tracks.append({
                    'track': t,
                    'face_emb': face_emb,
                    'body_emb': emb_b,
                    'person_center': person_center,
                    'anch_dist_boxes': anch_dist_boxes
                })
            else:
                no_face_tracks.append({
                    'track': t,
                    'body_emb': emb_b,
                    'person_center': person_center,
                    'anch_dist_boxes': anch_dist_boxes
                })

        # ── (E) 얼굴 있는 객체 우선 매칭 ─────────────────────────────────────────────
        matched_pids = set()   # 이 프레임에서 얼굴 매칭으로 할당된 pid 집합

        for entry in face_tracks:
            t = entry['track']
            face_emb = entry['face_emb']
            emb_b = entry['body_emb']
            person_center = entry['person_center']
            anch_dist_boxes = entry['anch_dist_boxes']

            # 얼굴 매칭 수행
            pid, sim = match_face(face_emb)
            reason = f"by FACE (sim={sim:.4f})"
            print(f"[Face] Frame {idx+1} | track_{t.track_id} → {pid} ({reason})")

            # 전신 임베딩 스무딩 방식으로 갱신 (항상 저장)
            if emb_b is not None:
                prev_body = person_gallery[pid]['body']
                if prev_body is not None:
                    updated = (1 - BODY_ALPHA) * prev_body + BODY_ALPHA * emb_b
                    person_gallery[pid]['body'] = updated / np.linalg.norm(updated)
                else:
                    person_gallery[pid]['body'] = emb_b

            # 히스토리 업데이트: 최소 정규화 거리(prev_dist_norm)만 기록
            min_dists_norm = {}
            for cls, boxes in anch_dist_boxes.items():
                dlist_norm = []
                for (bx1, by1, bx2, by2) in boxes:
                    c_x = (bx1 + bx2) // 2
                    c_y = (by1 + by2) // 2
                    d_raw = float(np.hypot(person_center[0] - c_x, person_center[1] - c_y))
                    area = float((bx2 - bx1) * (by2 - by1))
                    d_norm = d_raw / (np.sqrt(area) + 1e-6)
                    dlist_norm.append(d_norm)
                min_dists_norm[cls] = float(np.min(dlist_norm)) if dlist_norm else float('inf')

            person_gallery[pid]['history'].append({
                'frame': fname,
                'bbox': list(map(int, t.to_ltrb())),
                'anch_dist': min_dists_norm
            })

            # final_id에 매핑 및 시각화
            final_id[t.track_id] = pid
            matched_pids.add(pid)

            # 색상 및 박스 그리기
            color = pid2color[pid]
            l, t_top, r, b = t.to_ltrb()
            x1_t, y1_t, x2_t, y2_t = map(int, (l, t_top, r, b))
            cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), color, 2)
            cv2.putText(frame, f"{pid}", (x1_t, y1_t - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── (F) 얼굴 없는 객체 매칭 ────────────────────────────────────────────────
        # 후보 pid 리스트 = 이전 프레임까지 등장한 pid - matched_pids
        candidate_pids = [pid for pid in person_list if pid not in matched_pids]
        print(f"초기 후보군: {candidate_pids}")

        for entry in no_face_tracks:
            t = entry['track']
            emb_b = entry['body_emb']
            person_center = entry['person_center']
            anch_dist_boxes = entry['anch_dist_boxes']

            # 후보군 정보 출력
            print(f"▶ 소거 후보군 시작: {candidate_pids}")

            # 후보군이 비어 있으면 새로운 ID 할당
            if not candidate_pids:
                base_emb = emb_b if emb_b is not None else np.zeros(512, dtype=np.float32)
                pid = assign_person_id(base_emb, 'body')
                reason = "new ID (no candidates)"
                print(f"    후보 없음 → 새로운 ID: {pid}")
            else:
                # 제한된 후보군으로 매칭 수행
                pid_loc, score_combined = match_body_and_location(
                    emb_b, anch_dist_boxes, person_center, candidate_pids
                )
                if score_combined >= MATCH_THRESH:
                    pid = pid_loc
                    reason = f"by PCB+LOC (score={score_combined:.4f})"
                    print(f"    매칭된 PID: {pid} (score={score_combined:.4f})")
                else:
                    base_emb = emb_b if emb_b is not None else np.zeros(512, dtype=np.float32)
                    pid = assign_person_id(base_emb, 'body')
                    reason = "new ID (low score)"
                    print(f"    점수 낮음 → 새로운 ID: {pid}")

                # 매칭 후 후보군에서 소거
                if pid in candidate_pids:
                    candidate_pids.remove(pid)
                    print(f"    소거된 PID: {pid} → 남은 후보: {candidate_pids}")

                # 매칭 후 전신 임베딩 스무딩 방식으로 갱신 (항상 저장)
                if emb_b is not None:
                    prev_body = person_gallery[pid]['body']
                    if prev_body is not None:
                        updated = (1 - BODY_ALPHA) * prev_body + BODY_ALPHA * emb_b
                        person_gallery[pid]['body'] = updated / np.linalg.norm(updated)
                    else:
                        person_gallery[pid]['body'] = emb_b

            # 히스토리 업데이트: 최소 정규화 거리(prev_dist_norm)만 기록
            min_dists_norm = {}
            for cls, boxes in anch_dist_boxes.items():
                dlist_norm = []
                for (bx1, by1, bx2, by2) in boxes:
                    c_x = (bx1 + bx2) // 2
                    c_y = (by1 + by2) // 2
                    d_raw = float(np.hypot(person_center[0] - c_x, person_center[1] - c_y))
                    area = float((bx2 - bx1) * (by2 - by1))
                    d_norm = d_raw / (np.sqrt(area) + 1e-6)
                    dlist_norm.append(d_norm)
                min_dists_norm[cls] = float(np.min(dlist_norm)) if dlist_norm else float('inf')

            person_gallery[pid]['history'].append({
                'frame': fname,
                'bbox': list(map(int, t.to_ltrb())),
                'anch_dist': min_dists_norm
            })

            # final_id에 매핑 및 시각화
            final_id[t.track_id] = pid

            color = pid2color[pid]
            l, t_top, r, b = t.to_ltrb()
            x1_t, y1_t, x2_t, y2_t = map(int, (l, t_top, r, b))
            cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), color, 2)
            cv2.putText(frame, f"{pid}", (x1_t, y1_t - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print(f"Frame {idx+1} | track_{t.track_id} → {pid} ({reason})")

        # ── (G) person_gallery 상태 요약 출력 (주석 처리) ─────────────────────────────
        #print("\n--- [person_gallery 상태] ---")
        #for pid, info in person_gallery.items():
        #    has_face = "O" if info['face'] is not None else "X"
        #    has_body = "O" if info['body'] is not None else "X"
        #    history_len = len(info['history'])
        #    last_frame = info['history'][-1]['frame'] if history_len > 0 else "None"
        #    last_anchors = list(info['history'][-1]['anch_dist'].keys()) if history_len > 0 else []
        #    print(
        #        f"  • {pid}: face={has_face}, body={has_body}, "
        #        f"history_items={history_len}, last_frame={last_frame}, anchors={last_anchors}"
        #    )
        #print("-------------------------------\n")

        # ── (H) 프레임 결과 저장 ─────────────────────────────────────────────────
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, frame)

    # ── (10) 결과 영상 저장 (옵션) ───────────────────────────────────────────────
    if output_video:
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        if frame_files:
            h0, w0 = cv2.imread(os.path.join(output_dir, frame_files[0])).shape[:2]
            writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w0, h0))
            for f in frame_files:
                writer.write(cv2.imread(os.path.join(output_dir, f)))
            writer.release()

    print("Pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Identification Pipeline")
    parser.add_argument("--base_dir",       required=True,  help="기본 프레임/검출 경로")
    parser.add_argument("--frame_set_name", required=True,  help="프레임 세트 폴더명")
    parser.add_argument("--output_dir",     default="output_pipeline", help="결과 이미지 저장 폴더")
    parser.add_argument("--output_face_dir",default="output_faces",   help="얼굴 크롭 저장 폴더")
    parser.add_argument("--output_video",   default=None,     help="결과 영상 저장 경로 (예: result.mp4)")
    args = parser.parse_args()

    run_pipeline(args.base_dir, args.frame_set_name, args.output_dir, args.output_face_dir, args.output_video)
