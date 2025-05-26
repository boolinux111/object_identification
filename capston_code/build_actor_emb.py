import os
import numpy as np
import argparse
import install_dependencies

def build_actor_embeddings(actor_db_path, model_name="Facenet512", detector_backend="mtcnn"):
    """
    actor_db_path 디렉토리 내 하위 폴더(배우 이름)에서
    각 이미지 임베딩을 평균내어 {actor_name: avg_embedding} 형태의 dict 반환
    """

    from deepface import DeepFace

    actor_embeddings = {}
    for actor in os.listdir(actor_db_path):
        actor_folder = os.path.join(actor_db_path, actor)
        if not os.path.isdir(actor_folder) or actor.startswith('.'):
            continue

        embs = []
        for img in os.listdir(actor_folder):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(actor_folder, img)
            rep = DeepFace.represent(
                img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False
            )
            if rep and isinstance(rep, list) and 'embedding' in rep[0]:
                embs.append(np.array(rep[0]['embedding']))

        if embs:
            actor_embeddings[actor] = np.mean(embs, axis=0)
            print(f"  ▶ {actor}: {len(embs)}장 이미지로 임베딩 생성")
        else:
            print(f"  ⚠️ {actor}: 유효한 얼굴 이미지 없음")

    return actor_embeddings

if __name__ == "__main__":
    # 의존성 설치
    install_dependencies.install_all()

    # 명령줄 인자 파싱 및 임베딩 수행
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="actor_embeddings.npy")
    args = parser.parse_args()

    actor_embeddings = build_actor_embeddings(args.db_path)
    np.save(args.output, actor_embeddings)
    print(f"\n✅ 임베딩 저장 완료: {args.output}")
