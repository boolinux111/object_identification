import os
import subprocess

def run_command(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def install_all():
    # DeepFace git clone
    if not os.path.exists("deepface"):
        run_command("git clone https://github.com/serengil/deepface.git")

    # requirements.txt로 전체 패키지 설치
    run_command("pip install -r requirements.txt")

if __name__ == "__main__":
    print("🔧 의존성 설치를 시작합니다...")
    install_all()
    print("✅ 설치 완료.")