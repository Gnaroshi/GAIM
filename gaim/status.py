"""실험 폴더의 실제 기록만 읽어 진행 상황을 표시한다."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            # 실행 중 마지막 행을 쓰는 순간일 수 있다. 상태 화면에서만 미완료 행을 제외한다.
            continue
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="scripts/run_smoke.sh에 전달한 실행 이름")
    args = parser.parse_args()
    for phase, label in (("dev", "개발 평가"), ("train", "학습용 교란 생성"),
                         ("adapters", "네 조건 추가 학습"), ("replay", "학습 후 재평가")):
        folder = ROOT / "runs" / f"{args.run_name}_{phase}"
        if not folder.exists():
            print(f"{label}: 대기")
            continue
        if phase in ("dev", "train"):
            if not (folder / "run.json").exists():
                print(f"{label}: 준비 중")
                continue
            manifest = json.loads((folder / "run.json").read_text())
            count = sum(len(records(p)) for p in folder.glob("worker_*.jsonl"))
            expected = len(manifest["question_ids"]) * (1 + 3 * manifest["k"] + 3)
            state = "완료" if (folder / "COMPLETE").exists() else "진행 중 또는 모델 준비 중"
            print(f"{label}: {count}/{expected} 답변 기록 ({state})")
        elif phase == "adapters":
            for arm in ("clean", "random", "independent", "adaptive"):
                loss = records(folder / arm / "training_loss.jsonl")
                step = loss[-1]["step"] if loss else 0
                done = (folder / arm / "training_result.json").exists()
                print(f"{label}/{arm}: {step} 단계, 저장 완료={done}")
        else:
            count = sum(len(records(folder / f"{arm}.jsonl")) for arm in ("clean", "random", "independent", "adaptive"))
            print(f"{label}: {count} 답변 기록, 완료={(folder / 'COMPLETE').exists()}")
    exit_path = ROOT / "logs" / f"{args.run_name}.exit"
    if exit_path.exists():
        code = exit_path.read_text().strip()
        completed = (ROOT / "runs" / f"{args.run_name}_replay" / "COMPLETE").exists()
        if code == "0" and completed:
            print("전체 실행 정상 종료")
        else:
            print(f"전체 실행 미완료: 기록된 종료 코드 {code}; 각 단계 완료 여부와 실행 로그를 확인하세요.")


if __name__ == "__main__":
    main()
