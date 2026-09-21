"""Prepare a pinned, auditable MedQA sample without model calls or HF packages.

The Hugging Face repository is a third-party mirror, not the original authors'
repository. Its train/test labels are retained; our dev split is derived from
the mirror's train file. All medical text and answer keys remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid


DATASET_ID = "GBaker/MedQA-USMLE-4-options"
PINNED_REVISION = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
SOURCE_FILES = {
    "train": "phrases_no_exclude_train.jsonl",
    "test": "phrases_no_exclude_test.jsonl",
}
PINNED_SOURCE_SHA256 = {
    "train": "3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324",
    "test": "c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb",
}
SPLITS = ("train", "dev", "test")
SCHEMA_VERSION = 1
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
FetchBytes = Callable[[str], bytes]


class DataPreparationError(ValueError):
    """A source, configuration, or existing artifact cannot be trusted."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _records_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GAIM-MedQA-preparation/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataPreparationError(f"공개 원천을 내려받지 못했습니다: {url}: {exc}") from exc
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise DataPreparationError(f"원천 파일이 허용 크기를 초과했습니다: {url}")
    return payload


def _configuration(train_size: int, dev_size: int, test_size: int, seed: int, revision: str) -> dict[str, Any]:
    for name, value in (("train_size", train_size), ("dev_size", dev_size), ("test_size", test_size)):
        if type(value) is not int or value < 1:
            raise DataPreparationError(f"{name}는 1 이상의 정수여야 합니다.")
    if type(seed) is not int:
        raise DataPreparationError("seed는 정수여야 합니다.")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DataPreparationError("revision은 main이 아닌 40자리 소문자 commit SHA여야 합니다.")
    return {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "seed": seed,
        "train_size": train_size,
        "dev_size": dev_size,
        "test_size": test_size,
    }


def _parse_source(payload: bytes, split: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DataPreparationError(f"{split} 원천이 UTF-8이 아닙니다.") from exc
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        location = f"{split}:{index}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataPreparationError(f"JSONL 원천을 해석할 수 없습니다: {location}") from exc
        if not isinstance(raw, dict):
            raise DataPreparationError(f"문항이 객체가 아닙니다: {location}")
        question, options, answer = raw.get("question"), raw.get("options"), raw.get("answer_idx")
        if not isinstance(question, str) or not question.strip():
            raise DataPreparationError(f"question이 비어 있거나 문자열이 아닙니다: {location}")
        if not isinstance(options, dict) or set(options) != set("ABCD"):
            raise DataPreparationError(f"선택지는 정확히 A–D여야 합니다: {location}")
        if any(not isinstance(value, str) or not value.strip() for value in options.values()):
            raise DataPreparationError(f"선택지 내용이 비어 있거나 문자열이 아닙니다: {location}")
        if not isinstance(answer, str) or answer not in options:
            raise DataPreparationError(f"정답이 A–D 중 하나가 아닙니다: {location}")
        if "answer" in raw and raw["answer"] != options[answer]:
            raise DataPreparationError(f"원천의 정답 문자와 정답 본문이 다릅니다: {location}")
        # source_index는 원본 JSONL의 0부터 시작하는 행 번호다. 문자열은 정규화하지 않는다.
        records.append({
            "id": f"medqa-usmle4:{split}:{index:08d}",
            "question": question,
            "options": dict(options),
            "answer_idx": answer,
            "source_split": split,
            "source_index": index,
            "question_sha256": sha256_bytes(question.encode("utf-8")),
        })
    if not records:
        raise DataPreparationError(f"{split} 원천에 문항이 없습니다.")
    return records


def _same_question_content(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left[key] == right[key] for key in ("question", "options", "answer_idx"))


def _unique_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["question_sha256"], []).append(record)
    unique, conflicts = [], []
    duplicate_rows = 0
    for question_hash, group in grouped.items():
        # 원천에는 실제로 동일 질문에 서로 다른 선택지/정답을 붙인 행이 있다.
        # 의학적 정답을 임의로 결정하지 않고 충돌한 그룹의 모든 행을 제외한다.
        if any(not _same_question_content(group[0], row) for row in group[1:]):
            conflicts.append({
                "question_sha256": question_hash,
                "source_ids": [row["id"] for row in group],
                "reason": "conflicting options or answer keys for the same question; all group rows excluded",
            })
        else:
            unique.append(group[0])
            duplicate_rows += len(group) - 1
    return unique, duplicate_rows, conflicts


def _select_records(
    source_records: dict[str, list[dict[str, Any]]], configuration: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    unique_train, duplicate_train, conflicts_train = _unique_records(source_records["train"])
    unique_test, duplicate_test, conflicts_test = _unique_records(source_records["test"])
    test_by_hash = {record["question_sha256"]: record for record in unique_test}
    eligible_train = []
    overlap = 0
    # test 전체를 먼저 보호한다. 뽑히지 않은 test 문항도 train/dev로 유입시키지 않는다.
    for record in unique_train:
        test_record = test_by_hash.get(record["question_sha256"])
        if test_record is not None:
            if not _same_question_content(record, test_record):
                raise DataPreparationError(f"train/test 원천의 동일 문항이 상충합니다: {record['id']}")
            overlap += 1
        else:
            eligible_train.append(record)
    if len(eligible_train) < configuration["train_size"] + configuration["dev_size"]:
        raise DataPreparationError("중복 제거 후 train/dev에 필요한 원천 문항이 부족합니다.")
    if len(unique_test) < configuration["test_size"]:
        raise DataPreparationError("중복 제거 후 test에 필요한 원천 문항이 부족합니다.")

    def ranked(records: list[dict[str, Any]], namespace: str) -> list[dict[str, Any]]:
        # 난수 라이브러리 버전과 무관하게 seed + 출처 ID의 SHA 순서로 표본을 고정한다.
        return sorted(records, key=lambda row: (
            sha256_bytes(f"{configuration['seed']}|{namespace}|{row['id']}".encode("utf-8")), row["id"]
        ))

    ordered_train = ranked(eligible_train, "train-dev")
    dev_size, train_size = configuration["dev_size"], configuration["train_size"]
    # dev를 먼저 예약하므로 train_size를 늘려도 기존 dev가 학습으로 넘어가지 않는다.
    selected = {
        "dev": ordered_train[:dev_size],
        "train": ordered_train[dev_size:dev_size + train_size],
        "test": ranked(unique_test, "test")[:configuration["test_size"]],
    }
    selection = {
        "algorithm": "sha256(seed|namespace|source-id), dev reserved before train",
        "deduplication": "exact UTF-8 question SHA-256; identical duplicate rows retain first; conflicting groups excluded; test takes precedence",
        "source_index_definition": "zero-based original JSONL line index",
        "raw_counts": {split: len(rows) for split, rows in source_records.items()},
        "within_source_duplicates_removed": {"train": duplicate_train, "test": duplicate_test},
        "conflicting_question_groups_excluded": {"train": conflicts_train, "test": conflicts_test},
        "conflicting_source_rows_excluded": {
            "train": sum(len(group["source_ids"]) for group in conflicts_train),
            "test": sum(len(group["source_ids"]) for group in conflicts_test),
        },
        "train_questions_overlapping_source_test_removed": overlap,
        "eligible_counts": {"train_dev": len(eligible_train), "test": len(unique_test)},
        "selected_counts": {split: len(selected[split]) for split in SPLITS},
        "sampled_ids": {split: [row["id"] for row in selected[split]] for split in SPLITS},
    }
    return selected, selection


def _metadata_url(revision: str) -> str:
    return f"https://huggingface.co/api/datasets/{DATASET_ID}/revision/{revision}"


def _source_url(revision: str, split: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{SOURCE_FILES[split]}"


def _validate_metadata(payload: bytes, revision: str) -> dict[str, Any]:
    try:
        metadata = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DataPreparationError("HF 원천 metadata를 해석할 수 없습니다.") from exc
    if not isinstance(metadata, dict) or metadata.get("sha") != revision:
        raise DataPreparationError("HF API의 revision SHA가 요청한 고정 SHA와 다릅니다.")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise DataPreparationError("HF metadata에 유효한 파일 목록이 없습니다.")
    files = {item.get("rfilename") for item in siblings if isinstance(item, dict)}
    if not set(SOURCE_FILES.values()).issubset(files):
        raise DataPreparationError("고정된 revision에 필요한 train/test 원본 JSONL이 없습니다.")
    return metadata


def _build_artifacts(
    configuration: dict[str, Any], metadata_payload: bytes, sources: dict[str, bytes]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    revision = configuration["revision"]
    metadata = _validate_metadata(metadata_payload, revision)
    if revision == PINNED_REVISION:
        for split, expected in PINNED_SOURCE_SHA256.items():
            if sha256_bytes(sources[split]) != expected:
                raise DataPreparationError(f"{split} 원천 SHA-256이 확인된 고정 revision과 다릅니다.")
    source_records = {split: _parse_source(payload, split) for split, payload in sources.items()}
    selected, selection = _select_records(source_records, configuration)
    artifacts = {f"{split}.jsonl": _records_bytes(selected[split]) for split in SPLITS}
    artifacts.update({f"sources/{SOURCE_FILES[split]}": payload for split, payload in sources.items()})
    artifacts["sources/repository.json"] = metadata_payload
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration": configuration,
        "provenance": {
            "upstream_reference": "https://github.com/jind11/MedQA",
            "mirror_repository": f"https://huggingface.co/datasets/{DATASET_ID}",
            "source_kind": "third-party Hugging Face mirror",
            "metadata_url": _metadata_url(revision),
            "mirror_declared_license": metadata["cardData"].get("license") if isinstance(metadata.get("cardData"), dict) else None,
            "validation_is_official": False,
            "validation_origin": "dev is derived from the mirror train split, not an official validation split",
            "source_split_policy": "source_split always refers to the mirror source file, including dev records",
            "clinical_content_policy": "question, options, and answer_idx are copied unchanged; no model outputs used in sampling",
        },
        "sources": {
            split: {"path": f"sources/{SOURCE_FILES[split]}", "url": _source_url(revision, split),
                    "sha256": sha256_bytes(payload), "bytes": len(payload), "rows": len(source_records[split])}
            for split, payload in sources.items()
        },
        "selection": selection,
        "artifacts": {path: {"sha256": sha256_bytes(payload), "bytes": len(payload)} for path, payload in artifacts.items()},
    }
    return artifacts, manifest


def _read_existing(output_dir: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads((output_dir / "manifest.json").read_bytes())
        if not isinstance(manifest, dict) or manifest.get("configuration") != configuration:
            raise DataPreparationError("기존 데이터의 설정이 다릅니다. 새 output-dir 또는 명시적 --overwrite를 사용하세요.")
        # checksum 확인만으로 끝내지 않고 보관한 원본에서 표본과 manifest를 다시 계산한다.
        metadata_payload = (output_dir / "sources/repository.json").read_bytes()
        sources = {split: (output_dir / "sources" / name).read_bytes() for split, name in SOURCE_FILES.items()}
        artifacts, expected = _build_artifacts(configuration, metadata_payload, sources)
        if manifest != expected:
            raise DataPreparationError("기존 manifest가 보관된 원천과 표본 선정 규칙에 일치하지 않습니다.")
        for path, expected_payload in artifacts.items():
            if (output_dir / path).read_bytes() != expected_payload:
                raise DataPreparationError(f"기존 파일이 원천/checksum과 일치하지 않습니다: {path}")
        return manifest
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, DataPreparationError):
            raise
        raise DataPreparationError(f"기존 데이터 검증에 실패했습니다: {exc}") from exc


def _check_overwrite_contents(output_dir: Path) -> None:
    managed = {"manifest.json", "sources", *(f"{split}.jsonl" for split in SPLITS)}
    unknown = {entry.name for entry in output_dir.iterdir()} - managed
    source_dir = output_dir / "sources"
    for entry in output_dir.iterdir():
        if entry.name != "sources" and entry.name in managed and (entry.is_symlink() or not entry.is_file()):
            raise DataPreparationError(f"관리 파일 위치에 다른 종류의 항목이 있어 덮어쓰지 않습니다: {entry.name}")
    if source_dir.is_symlink():
        raise DataPreparationError("sources 심볼릭 링크가 있는 기존 디렉터리는 덮어쓰지 않습니다.")
    if source_dir.exists() and not source_dir.is_dir():
        raise DataPreparationError("sources 위치가 디렉터리가 아니므로 덮어쓰지 않습니다.")
    if source_dir.is_dir():
        unknown.update(f"sources/{entry.name}" for entry in source_dir.iterdir()
                       if entry.name not in {*SOURCE_FILES.values(), "repository.json"})
        if any(entry.is_symlink() or not entry.is_file() for entry in source_dir.iterdir()):
            raise DataPreparationError("sources 내 항목이 일반 파일이 아니므로 덮어쓰지 않습니다.")
    if unknown:
        raise DataPreparationError(f"데이터 준비 모듈이 관리하지 않는 파일이 있어 덮어쓰지 않습니다: {sorted(unknown)}")


def prepare_dataset(
    output_dir: str | Path, *, train_size: int = 100, dev_size: int = 20,
    test_size: int = 100, seed: int = 20260921, revision: str = PINNED_REVISION,
    overwrite: bool = False, fetch_bytes: FetchBytes | None = None,
) -> dict[str, Any]:
    """Create train/dev/test JSONL + provenance manifest; reuse only after verification.

    ``fetch_bytes`` is an injectable public-download function for offline tests.
    Existing matching artifacts are verified entirely offline. No model calls
    are made. ``overwrite`` explicitly replaces this module's managed dataset.
    """
    configuration = _configuration(train_size, dev_size, test_size, seed, revision)
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise DataPreparationError("output-dir는 심볼릭 링크가 아닌 디렉터리여야 합니다.")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            return _read_existing(output, configuration)
        _check_overwrite_contents(output)

    fetch = fetch_bytes or _fetch_bytes
    metadata_payload = fetch(_metadata_url(revision))
    _validate_metadata(metadata_payload, revision)
    sources = {split: fetch(_source_url(revision, split)) for split in SOURCE_FILES}
    artifacts, manifest = _build_artifacts(configuration, metadata_payload, sources)
    # 모두 검증한 뒤 임시 디렉터리를 교체하여 실패한 다운로드가 기존 표본을 훼손하지 않게 한다.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-prepare-", dir=output.parent))
    backup: Path | None = None
    try:
        for path, payload in {**artifacts, "manifest.json": _json_bytes(manifest)}.items():
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        if output.exists():
            if any(output.iterdir()) and not overwrite:
                raise DataPreparationError("준비 중 output-dir가 변경되었습니다. 기존 자료를 덮어쓰지 않았습니다.")
            _check_overwrite_contents(output)
            backup = output.with_name(f".{output.name}-previous-{uuid.uuid4().hex}")
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except OSError:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="고정된 공개 MedQA 원천에서 검증 가능한 표본을 준비합니다.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/medqa"))
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--dev-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--revision", default=PINNED_REVISION, help="원천의 고정된 40자리 commit SHA")
    parser.add_argument("--overwrite", action="store_true", help="이 모듈이 관리하는 기존 데이터만 명시적으로 교체")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset(**vars(args))
    except (DataPreparationError, OSError) as exc:
        print(f"데이터 준비 오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir.absolute()),
        "revision": manifest["configuration"]["revision"],
        "counts": manifest["selection"]["selected_counts"],
        "dev_origin": "mirror train에서 분리한 자체 개발 세트",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
