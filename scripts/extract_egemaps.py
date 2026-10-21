#!/usr/bin/env python3
"""Extract openSMILE eGeMAPSv02 functionals from DAIC-WOZ WAV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import opensmile
import pandas as pd


def participant_id_from_audio(path: Path) -> str:
    match = re.match(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(
            f"Cannot infer a DAIC-WOZ participant ID from audio filename: {path.name}"
        )
    return match.group(1)


def normalize_participant_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Participant IDs in the labels file must not be empty.")
    participant_id = str(value).strip().strip("'\"")
    if re.fullmatch(r"\d+\.0+", participant_id):
        participant_id = participant_id.split(".", maxsplit=1)[0]
    if not participant_id:
        raise ValueError("Participant IDs in the labels file must not be empty.")
    return participant_id


def load_labels(
    labels_path: Path,
    id_column: str,
    audio_name_column: str,
    label_column: str,
    score_column: str | None,
    positive_threshold: float,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    required_columns = [id_column, audio_name_column]
    missing_columns = [column for column in required_columns if column not in labels]
    if missing_columns:
        raise ValueError(
            f"Required column(s) {', '.join(repr(col) for col in missing_columns)} "
            f"were not found in {labels_path}. "
            f"Available columns: {', '.join(labels.columns)}"
        )
    labels["participant_id"] = labels[id_column].map(normalize_participant_id)
    labels["audio_file"] = labels[audio_name_column].astype("string").str.strip()
    missing_audio_names = labels["audio_file"].isna() | labels["audio_file"].eq("")
    if missing_audio_names.any():
        raise ValueError(
            f"{labels_path} has {int(missing_audio_names.sum())} row(s) with no "
            f"audio filename in {audio_name_column!r}."
        )

    if score_column:
        if score_column not in labels.columns:
            raise ValueError(
                f"Score column {score_column!r} was not found in {labels_path}. "
                f"Available columns: {', '.join(labels.columns)}"
            )
        scores = pd.to_numeric(labels[score_column], errors="raise")
        labels["label"] = scores.ge(positive_threshold).astype("Int64")
        labels.loc[scores.isna(), "label"] = pd.NA
    else:
        if label_column not in labels.columns:
            raise ValueError(
                f"Label column {label_column!r} was not found in {labels_path}. "
                "Use --score-column to derive labels from a PHQ score column."
            )
        label_values = pd.to_numeric(labels[label_column], errors="raise")
        invalid_labels = label_values.dropna().loc[~label_values.dropna().isin([0, 1])]
        if not invalid_labels.empty:
            raise ValueError(
                f"{label_column!r} must contain binary 0/1 values. "
                "For PHQ scores, pass --score-column and --positive-threshold."
            )
        labels["label"] = label_values.astype("Int64")

    duplicate_rows = labels["audio_file"].duplicated(keep=False)
    if duplicate_rows.any():
        duplicate_groups = labels.loc[duplicate_rows].groupby("audio_file", sort=False)
        conflicting_audio = [
            audio_file
            for audio_file, group in duplicate_groups
            if group["participant_id"].nunique(dropna=False) > 1
            or group["label"].nunique(dropna=False) > 1
        ]
        if conflicting_audio:
            examples = ", ".join(conflicting_audio[:10])
            raise ValueError(
                "The labels file maps duplicate audio file(s) to conflicting "
                f"participant IDs or labels: {examples}."
            )

        duplicate_count = len(labels) - labels["audio_file"].nunique()
        labels = labels.drop_duplicates(subset="audio_file", keep="first")
        print(
            f"Collapsed {duplicate_count} duplicate label row(s) with identical "
            "participant IDs and labels."
        )

    return labels[["participant_id", "audio_file", "label"]].reset_index(drop=True)


def resolve_audio_file(audio_name: str, audio_dir: Path, files_by_name: dict[str, Path]) -> Path:
    candidate = Path(audio_name).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".wav")
    if not candidate.is_absolute():
        candidate = audio_dir / candidate
    if candidate.is_file():
        return candidate

    match = files_by_name.get(candidate.name.casefold())
    if match is not None:
        return match
    raise FileNotFoundError(
        f"Could not find WAV {audio_name!r} under {audio_dir}. "
        "Check that --audio-dir points to the directory containing the "
        "segment WAVs named by the labels CSV."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one eGeMAPSv02 functionals row per DAIC-WOZ WAV file. "
            "When labels are supplied, process and label only the WAVs named "
            "in that segment-level CSV."
        )
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing DAIC-WOZ WAV files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/daicwoz_egemaps.csv"),
        help="Combined output CSV path.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="CSV mapping each audio segment to a participant and depression label.",
    )
    parser.add_argument(
        "--audio-name-column",
        default="wavname",
        help="Audio filename column in the labels CSV (default: wavname).",
    )
    parser.add_argument(
        "--id-column",
        default="Participant_ID",
        help="Participant ID column in the labels CSV (default: Participant_ID).",
    )
    parser.add_argument(
        "--label-column",
        default="PHQ8_Binary",
        help="Binary 0/1 label column (default: PHQ8_Binary).",
    )
    parser.add_argument(
        "--score-column",
        help="PHQ-8 score column; if provided, label is 1 when score >= threshold.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=10,
        help="Positive class threshold used with --score-column (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N WAVs/label rows (useful for a short smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")
    if not args.audio_dir.is_dir():
        raise NotADirectoryError(f"Audio directory does not exist: {args.audio_dir}")

    discovered_audio_files = sorted(
        path
        for path in args.audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not discovered_audio_files:
        raise FileNotFoundError(f"No WAV files found under {args.audio_dir}")

    labels = None
    files_by_name: dict[str, Path] = {}
    for path in discovered_audio_files:
        key = path.name.casefold()
        if key in files_by_name:
            raise ValueError(
                f"Multiple WAV files named {path.name!r} were found under "
                f"{args.audio_dir}. Point --audio-dir at a folder with unique "
                "segment filenames."
            )
        files_by_name[key] = path

    if args.labels:
        if not args.labels.is_file():
            raise FileNotFoundError(f"Labels CSV does not exist: {args.labels}")
        labels = load_labels(
            args.labels,
            args.id_column,
            args.audio_name_column,
            args.label_column,
            args.score_column,
            args.positive_threshold,
        )
        if args.limit is not None:
            labels = labels.head(args.limit)
        audio_files = [
            resolve_audio_file(name, args.audio_dir, files_by_name)
            for name in labels["audio_file"]
        ]
    else:
        audio_files = discovered_audio_files
        if args.limit is not None:
            audio_files = audio_files[: args.limit]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    rows = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] {audio_path.name}", flush=True)
        extracted = smile.process_file(str(audio_path))
        if len(extracted) != 1:
            raise ValueError(
                f"Expected one functionals row for {audio_path}, got {len(extracted)}."
            )
        row = extracted.iloc[0].to_dict()
        row["participant_id"] = (
            labels.iloc[index - 1]["participant_id"]
            if labels is not None
            else participant_id_from_audio(audio_path)
        )
        row["audio_file"] = audio_path.name
        if labels is not None:
            row["label"] = labels.iloc[index - 1]["label"]
        rows.append(row)

    features = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(
        f"Saved {len(features)} audio rows and {len(features.columns)} columns "
        f"to {args.output}"
    )
if __name__ == "__main__":
    main()
