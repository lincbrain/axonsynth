"""Strict configuration support for the fixed 12-patch paper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from string import Formatter
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "lsm_12patch.json"
REQUIRED_ENVIRONMENT_VARIABLES = (
    "DATA_ROOT",
    "OUTPUT_ROOT",
    "THREE_CLASS_CHECKPOINT",
    "BINARY_CHECKPOINT",
)
DOMAINS = ("human", "macaque")
SPLITS = ("calibration", "heldout")
METHOD_NAMES = (
    "three_class_v2",
    "binary_baseline",
    "intensity_threshold",
    "frangi_bright",
)
LEARNED_METHOD_NAMES = METHOD_NAMES[:2]
BASELINE_METHOD_NAMES = METHOD_NAMES[2:]

# The full records, rather than only their IDs, are fixed so a manifest cannot
# silently relabel a calibration patch or acquisition domain.
EXPECTED_PATCH_RECORDS = (
    (
        "human_NEFH/Human_NEFH_GM",
        "human_NEFH",
        "Human_NEFH_GM",
        "human",
        "calibration",
    ),
    (
        "human_NEFH/Human_NEFH_WM",
        "human_NEFH",
        "Human_NEFH_WM",
        "human",
        "heldout",
    ),
    (
        "human_NEFH/Human_NEFH_transition",
        "human_NEFH",
        "Human_NEFH_transition",
        "human",
        "heldout",
    ),
    (
        "human_NEFL/human_NEFL_GM_1",
        "human_NEFL",
        "human_NEFL_GM_1",
        "human",
        "heldout",
    ),
    (
        "human_NEFL/human_NEFL_GM_2",
        "human_NEFL",
        "human_NEFL_GM_2",
        "human",
        "heldout",
    ),
    (
        "human_NEFL/human_NEFL_White-Gray_Transition",
        "human_NEFL",
        "human_NEFL_White-Gray_Transition",
        "human",
        "heldout",
    ),
    (
        "macaque_NEFH/macaque_NEFH_GM",
        "macaque_NEFH",
        "macaque_NEFH_GM",
        "macaque",
        "heldout",
    ),
    (
        "macaque_NEFH/macaque_NEFH_WM",
        "macaque_NEFH",
        "macaque_NEFH_WM",
        "macaque",
        "heldout",
    ),
    (
        "macaque_NEFH/macaque_NEFH_transition",
        "macaque_NEFH",
        "macaque_NEFH_transition",
        "macaque",
        "heldout",
    ),
    (
        "macaque_PV/macaque_PV_WM_1",
        "macaque_PV",
        "macaque_PV_WM_1",
        "macaque",
        "heldout",
    ),
    (
        "macaque_PV/macaque_PV_WM_2",
        "macaque_PV",
        "macaque_PV_WM_2",
        "macaque",
        "calibration",
    ),
    (
        "macaque_PV/macaque_PV_White-Gray_Transition",
        "macaque_PV",
        "macaque_PV_White-Gray_Transition",
        "macaque",
        "heldout",
    ),
)
EXPECTED_PATCH_IDS = tuple(record[0] for record in EXPECTED_PATCH_RECORDS)

FIXED_THRESHOLDS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "three_class_v2": MappingProxyType({"human": 0.94, "macaque": 0.88}),
        "binary_baseline": MappingProxyType({"human": 0.95, "macaque": 0.95}),
        "intensity_threshold": MappingProxyType(
            {"human": 0.47, "macaque": 0.46}
        ),
        "frangi_bright": MappingProxyType({"human": 0.05, "macaque": 0.01}),
    }
)

_EXPECTED_METHOD_SETTINGS = {
    "three_class_v2": {
        "kind": "learned",
        "segmentation_mode": "three_class_shell_interior",
        "checkpoint_environment": "THREE_CLASS_CHECKPOINT",
        "threshold_selection_metric": "mean_target_informed_neighbor_corrected_dice_on_domain_calibration_patch",
        "baseline": None,
    },
    "binary_baseline": {
        "kind": "learned",
        "segmentation_mode": "binary",
        "checkpoint_environment": "BINARY_CHECKPOINT",
        "threshold_selection_metric": "mean_target_informed_neighbor_corrected_dice_on_domain_calibration_patch",
        "baseline": None,
    },
    "intensity_threshold": {
        "kind": "baseline",
        "segmentation_mode": None,
        "checkpoint_environment": None,
        "threshold_selection_metric": "mean_target_informed_neighbor_corrected_dice_on_domain_calibration_patch",
        "baseline": "intensity_threshold",
    },
    "frangi_bright": {
        "kind": "baseline",
        "segmentation_mode": None,
        "checkpoint_environment": None,
        "threshold_selection_metric": "mean_raw_dice_on_domain_calibration_patch",
        "baseline": "frangi",
    },
}

_EXPECTED_FRANGI_SETTINGS = {
    "input": "normalized_intensity",
    "ridge_polarity": "bright",
    "black_ridges": False,
    "mode": "reflect",
    "gamma": None,
    "selection_metric": "mean_raw_dice_on_calibration_patches",
    "reporting_scope": "domain_specific",
    "accepted_parameters": {
        "global": {
            "alpha": 0.25,
            "beta": 0.5,
            "sigmas": [3, 5, 7, 9],
            "threshold": 0.03,
            "calibration_mean_raw_dice": 0.46314261842083926,
            "n_calibration_patches": 2,
        },
        "by_domain": {
            "human": {
                "alpha": 0.25,
                "beta": 0.5,
                "sigmas": [3, 4, 5, 6, 7],
                "threshold": 0.05,
                "calibration_mean_raw_dice": 0.48971089485627695,
                "n_calibration_patches": 1,
            },
            "macaque": {
                "alpha": 0.25,
                "beta": 0.33,
                "sigmas": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "threshold": 0.01,
                "calibration_mean_raw_dice": 0.46716461780183444,
                "n_calibration_patches": 1,
            },
        },
    },
}

REQUIRED_PATH_TEMPLATES = (
    "raw_patch",
    "target_mask",
    "prediction_dir",
    "prediction_score",
    "prediction_mask",
    "learned_metadata",
    "baseline_metadata",
    "metrics_csv",
    "metrics_json",
    "macro_metrics_csv",
    "macro_metrics_json",
    "depth_metrics_csv",
    "depth_metrics_json",
    "raw_error_overlay",
    "overlay_manifest",
    "packet_dir",
    "packet_table",
    "packet_manifest",
    "packet_markdown",
)

_DATA_TEMPLATES = {"raw_patch", "target_mask"}
_FORMAT_FIELDS = {"dataset", "patch", "patch_id", "domain", "split", "method"}
_ENVIRONMENT_TOKEN = re.compile(r"\$\{([^{}]+)\}")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class PatchConfig:
    """One immutable member of the paper patch allowlist."""

    id: str
    dataset: str
    patch: str
    domain: str
    split: str


@dataclass(frozen=True)
class MethodConfig:
    """Fixed inference or baseline settings for one reported method."""

    name: str
    kind: str
    thresholds: Mapping[str, float]
    segmentation_mode: str | None = None
    checkpoint: Path | None = None
    baseline: str | None = None
    threshold_selection_metric: str = ""

    def threshold_for(self, domain: str) -> float:
        try:
            return self.thresholds[domain]
        except KeyError as error:
            raise ValueError(f"Method {self.name!r} has no threshold for {domain!r}") from error


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved and validated canonical pipeline configuration."""

    source: Path
    data_root: Path
    output_root: Path
    environment: Mapping[str, str]
    path_templates: Mapping[str, str]
    patches: tuple[PatchConfig, ...]
    methods: Mapping[str, MethodConfig]
    neighbor_rounds: int
    depth_axis: int
    depth_slabs: int

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(self.methods)

    def get_patch(self, patch: PatchConfig | str) -> PatchConfig:
        if isinstance(patch, PatchConfig):
            if patch not in self.patches:
                raise ValueError(f"Patch {patch.id!r} is not in the fixed allowlist")
            return patch
        for candidate in self.patches:
            if candidate.id == patch:
                return candidate
        raise ValueError(f"Patch {patch!r} is not in the fixed allowlist")

    def get_method(self, method: MethodConfig | str) -> MethodConfig:
        if isinstance(method, MethodConfig):
            configured = self.methods.get(method.name)
            if configured != method:
                raise ValueError(f"Method {method.name!r} is not canonical")
            return method
        try:
            return self.methods[method]
        except KeyError as error:
            raise ValueError(f"Method {method!r} is not configured") from error

    def format_path(
        self,
        template_name: str,
        *,
        patch: PatchConfig | str | None = None,
        method: MethodConfig | str | None = None,
    ) -> Path:
        """Safely format one named path and keep it under its declared root."""

        try:
            template = self.path_templates[template_name]
        except KeyError as error:
            raise ValueError(f"Unknown path template {template_name!r}") from error

        patch_config = self.get_patch(patch) if patch is not None else None
        method_config = self.get_method(method) if method is not None else None
        values: dict[str, str] = {}
        if patch_config is not None:
            values.update(
                {
                    "dataset": patch_config.dataset,
                    "patch": patch_config.patch,
                    "patch_id": patch_config.id,
                    "domain": patch_config.domain,
                    "split": patch_config.split,
                }
            )
        if method_config is not None:
            values["method"] = method_config.name

        rendered, environment_names = _format_template(
            template,
            values,
            self.environment,
            context=f"path template {template_name!r}",
        )
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Path template {template_name!r} did not produce an absolute path")
        resolved = path.resolve()

        expected_environment = (
            "DATA_ROOT" if template_name in _DATA_TEMPLATES else "OUTPUT_ROOT"
        )
        if environment_names != {expected_environment}:
            raise ValueError(
                f"Path template {template_name!r} must use only "
                f"${{{expected_environment}}}"
            )
        root = self.data_root if expected_environment == "DATA_ROOT" else self.output_root
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Path template {template_name!r} escapes configured root {root}"
            )
        return resolved

    # ``path`` is the concise spelling used by pipeline stages.
    path = format_path


def _required_mapping(mapping: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} requires a mapping {key!r}")
    return value


def _required_nonempty_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty string {key!r}")
    return value


def _resolved_environment(environ: Mapping[str, str]) -> Mapping[str, str]:
    resolved: dict[str, str] = {}
    for name in REQUIRED_ENVIRONMENT_VARIABLES:
        raw_value = environ.get(name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Required environment variable {name} is not set")
        if "\x00" in raw_value:
            raise ValueError(f"Environment variable {name} contains a NUL byte")
        path = Path(raw_value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Environment variable {name} must be an absolute path")
        resolved[name] = str(path.resolve())
    return MappingProxyType(resolved)


def _protect_environment_tokens(
    template: str,
    context: str,
) -> tuple[str, dict[str, str]]:
    tokens: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in REQUIRED_ENVIRONMENT_VARIABLES:
            raise ValueError(f"{context} uses unsupported environment variable {name!r}")
        sentinel = f"__PAPER_ENV_{len(tokens)}__"
        tokens[sentinel] = name
        return sentinel

    protected = _ENVIRONMENT_TOKEN.sub(replace, template)
    if "${" in protected:
        raise ValueError(f"{context} contains malformed environment substitution")
    return protected, tokens


def _format_template(
    template: str,
    values: Mapping[str, str],
    environment: Mapping[str, str],
    *,
    context: str,
) -> tuple[str, set[str]]:
    protected, tokens = _protect_environment_tokens(template, context)
    formatter = Formatter()
    fields: set[str] = set()
    try:
        parsed = tuple(formatter.parse(protected))
    except ValueError as error:
        raise ValueError(f"{context} has invalid braces: {error}") from error

    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name not in _FORMAT_FIELDS:
            raise ValueError(f"{context} uses unsupported field {field_name!r}")
        if format_spec or conversion:
            raise ValueError(f"{context} cannot use conversions or format specifications")
        fields.add(field_name)
    missing = sorted(fields - values.keys())
    if missing:
        raise ValueError(f"{context} is missing formatting value(s): {', '.join(missing)}")

    try:
        rendered = formatter.vformat(protected, (), dict(values))
    except (KeyError, ValueError) as error:
        raise ValueError(f"Could not format {context}: {error}") from error
    environment_names = set(tokens.values())
    for sentinel, name in tokens.items():
        rendered = rendered.replace(sentinel, environment[name])
    return rendered, environment_names


def _resolve_environment_path(
    value: str,
    environment: Mapping[str, str],
    *,
    expected_name: str,
    context: str,
) -> Path:
    rendered, names = _format_template(value, {}, environment, context=context)
    if names != {expected_name}:
        raise ValueError(f"{context} must resolve from ${{{expected_name}}}")
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{context} must resolve to an absolute path")
    return path.resolve()


def _validate_patches(document: Mapping[str, Any]) -> tuple[PatchConfig, ...]:
    raw_patches = document.get("patches")
    if not isinstance(raw_patches, list):
        raise ValueError("Config requires a 'patches' list")

    records: list[tuple[str, str, str, str, str]] = []
    patches: list[PatchConfig] = []
    expected_keys = ("id", "dataset", "patch", "domain", "split")
    for index, raw_patch in enumerate(raw_patches):
        if not isinstance(raw_patch, Mapping):
            raise ValueError(f"Patch at index {index} must be a mapping")
        if tuple(raw_patch) != expected_keys:
            raise ValueError(
                f"Patch at index {index} must contain the ordered canonical fields"
            )
        record = tuple(
            _required_nonempty_string(raw_patch, key, f"Patch at index {index}")
            for key in ("id", "dataset", "patch", "domain", "split")
        )
        records.append(record)  # type: ignore[arg-type]
        patches.append(PatchConfig(*record))

    if tuple(records) != EXPECTED_PATCH_RECORDS:
        raise ValueError("Config patches must exactly match the fixed 12-patch allowlist")
    if sum(record[4] == "calibration" for record in records) != 2:
        raise ValueError("The fixed patch set must contain exactly two calibration patches")
    if sum(record[4] == "heldout" for record in records) != 10:
        raise ValueError("The fixed patch set must contain exactly ten heldout patches")
    for _, dataset, patch, domain, split in records:
        if not _PATH_COMPONENT.fullmatch(dataset) or not _PATH_COMPONENT.fullmatch(patch):
            raise ValueError("Dataset and patch names must be safe path components")
        if domain not in DOMAINS or split not in SPLITS:
            raise ValueError("Patch domain or split is not supported")
    return tuple(patches)


def _validate_methods(
    document: Mapping[str, Any],
    environment: Mapping[str, str],
) -> Mapping[str, MethodConfig]:
    raw_methods = _required_mapping(document, "methods", "Config")
    if tuple(raw_methods) != METHOD_NAMES:
        raise ValueError(
            "Config methods must be ordered exactly as: " + ", ".join(METHOD_NAMES)
        )

    methods: dict[str, MethodConfig] = {}
    for name in METHOD_NAMES:
        raw_method = raw_methods[name]
        if not isinstance(raw_method, Mapping):
            raise ValueError(f"Method {name!r} must be a mapping")
        expected = _EXPECTED_METHOD_SETTINGS[name]
        kind = _required_nonempty_string(raw_method, "kind", f"Method {name!r}")
        if kind != expected["kind"]:
            raise ValueError(f"Method {name!r} must have kind {expected['kind']!r}")

        raw_thresholds = _required_mapping(raw_method, "thresholds", f"Method {name!r}")
        if tuple(raw_thresholds) != DOMAINS:
            raise ValueError(f"Method {name!r} must define human and macaque thresholds")
        try:
            thresholds = {domain: float(raw_thresholds[domain]) for domain in DOMAINS}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Method {name!r} thresholds must be numeric") from error
        if thresholds != dict(FIXED_THRESHOLDS[name]):
            raise ValueError(f"Method {name!r} thresholds differ from the fixed values")
        threshold_selection_metric = _required_nonempty_string(
            raw_method, "threshold_selection_metric", f"Method {name!r}"
        )
        if threshold_selection_metric != expected["threshold_selection_metric"]:
            raise ValueError(
                f"Method {name!r} has the wrong threshold selection metric"
            )

        checkpoint: Path | None = None
        segmentation_mode: str | None = None
        baseline: str | None = None
        if kind == "learned":
            segmentation_mode = _required_nonempty_string(
                raw_method, "segmentation_mode", f"Method {name!r}"
            )
            if segmentation_mode != expected["segmentation_mode"]:
                raise ValueError(f"Method {name!r} has the wrong segmentation mode")
            checkpoint_value = _required_nonempty_string(
                raw_method, "checkpoint", f"Method {name!r}"
            )
            checkpoint = _resolve_environment_path(
                checkpoint_value,
                environment,
                expected_name=str(expected["checkpoint_environment"]),
                context=f"Method {name!r} checkpoint",
            )
        else:
            baseline = _required_nonempty_string(
                raw_method, "baseline", f"Method {name!r}"
            )
            if baseline != expected["baseline"]:
                raise ValueError(f"Method {name!r} does not select the accepted baseline")

        methods[name] = MethodConfig(
            name=name,
            kind=kind,
            thresholds=MappingProxyType(thresholds),
            segmentation_mode=segmentation_mode,
            checkpoint=checkpoint,
            baseline=baseline,
            threshold_selection_metric=threshold_selection_metric,
        )

    return MappingProxyType(methods)


def _validate_templates(document: Mapping[str, Any]) -> Mapping[str, str]:
    raw_templates = _required_mapping(document, "path_templates", "Config")
    if tuple(raw_templates) != REQUIRED_PATH_TEMPLATES:
        raise ValueError(
            "Config path_templates must contain the canonical ordered template set"
        )
    templates: dict[str, str] = {}
    for name in REQUIRED_PATH_TEMPLATES:
        template = _required_nonempty_string(raw_templates, name, "Config path_templates")
        protected, _ = _protect_environment_tokens(template, f"path template {name!r}")
        try:
            parsed = tuple(Formatter().parse(protected))
        except ValueError as error:
            raise ValueError(f"path template {name!r} has invalid braces: {error}") from error
        for _, field_name, format_spec, conversion in parsed:
            if field_name is not None and field_name not in _FORMAT_FIELDS:
                raise ValueError(
                    f"path template {name!r} uses unsupported field {field_name!r}"
                )
            if field_name is not None and (format_spec or conversion):
                raise ValueError(
                    f"path template {name!r} cannot use formatting operations"
                )
        templates[name] = template
    return MappingProxyType(templates)


def _validate_evaluation(document: Mapping[str, Any]) -> tuple[int, int, int]:
    evaluation = _required_mapping(document, "evaluation", "Config")
    correction = _required_mapping(
        evaluation, "target_informed_neighbor_correction", "Config evaluation"
    )
    if correction.get("enabled") is not True:
        raise ValueError("Target-informed neighbor correction must be enabled")
    if correction.get("connectivity") != "face-6":
        raise ValueError("Target-informed correction connectivity must be face-6")
    rounds = correction.get("neighbor_rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds != 2:
        raise ValueError("Target-informed correction must use exactly two rounds")

    depth = _required_mapping(evaluation, "depth_analysis", "Config evaluation")
    axis = depth.get("axis")
    slabs = depth.get("slabs")
    if isinstance(axis, bool) or not isinstance(axis, int) or axis != 0:
        raise ValueError("Depth analysis axis must be 0")
    if isinstance(slabs, bool) or not isinstance(slabs, int) or slabs != 8:
        raise ValueError("Depth analysis must use exactly 8 slabs")
    return rounds, axis, slabs


def _validate_frangi(document: Mapping[str, Any]) -> None:
    frangi = _required_mapping(document, "frangi", "Config")
    if dict(frangi) != _EXPECTED_FRANGI_SETTINGS:
        raise ValueError("Config Frangi settings differ from the fixed accepted recipe")


def load_pipeline_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    environ: Mapping[str, str] | None = None,
) -> PipelineConfig:
    """Load the canonical JSON and resolve all four site-specific paths."""

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".json":
        raise ValueError("The canonical paper config must be JSON")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {source}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("Canonical config must contain a mapping at its root")
    if document.get("schema_version") != 3:
        raise ValueError("Canonical config schema_version must be 3")

    environment = _resolved_environment(os.environ if environ is None else environ)
    roots = _required_mapping(document, "roots", "Config")
    if tuple(roots) != ("data", "output"):
        raise ValueError("Config roots must contain only data and output")
    data_root = _resolve_environment_path(
        _required_nonempty_string(roots, "data", "Config roots"),
        environment,
        expected_name="DATA_ROOT",
        context="data root",
    )
    output_root = _resolve_environment_path(
        _required_nonempty_string(roots, "output", "Config roots"),
        environment,
        expected_name="OUTPUT_ROOT",
        context="output root",
    )

    patches = _validate_patches(document)
    methods = _validate_methods(document, environment)
    templates = _validate_templates(document)
    _validate_frangi(document)
    neighbor_rounds, depth_axis, depth_slabs = _validate_evaluation(document)
    config = PipelineConfig(
        source=source,
        data_root=data_root,
        output_root=output_root,
        environment=environment,
        path_templates=templates,
        patches=patches,
        methods=methods,
        neighbor_rounds=neighbor_rounds,
        depth_axis=depth_axis,
        depth_slabs=depth_slabs,
    )

    # Eagerly exercise every template with representative records, including
    # all prediction and overlay paths whose uniqueness is required downstream.
    for name in REQUIRED_PATH_TEMPLATES:
        template = templates[name]
        protected, _ = _protect_environment_tokens(template, f"path template {name!r}")
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(protected)
            if field_name is not None
        }
        patch = patches[0] if fields & {"dataset", "patch", "patch_id", "domain", "split"} else None
        method = methods[METHOD_NAMES[0]] if "method" in fields else None
        config.path(name, patch=patch, method=method)

    for template_name in ("raw_patch", "target_mask"):
        paths = {config.path(template_name, patch=patch) for patch in patches}
        if len(paths) != len(patches):
            raise ValueError(f"{template_name} template does not produce 12 unique paths")

    for template_name in (
        "prediction_dir",
        "prediction_score",
        "prediction_mask",
        "learned_metadata",
        "baseline_metadata",
        "raw_error_overlay",
    ):
        paths = {
            config.path(template_name, patch=patch, method=method)
            for patch in patches
            for method in METHOD_NAMES
        }
        if len(paths) != len(patches) * len(METHOD_NAMES):
            raise ValueError(f"{template_name} template does not produce 48 unique paths")

    for patch in patches:
        for method in METHOD_NAMES:
            prediction_dir = config.path("prediction_dir", patch=patch, method=method)
            for template_name in (
                "prediction_score",
                "prediction_mask",
                "learned_metadata",
                "baseline_metadata",
            ):
                if config.path(template_name, patch=patch, method=method).parent != prediction_dir:
                    raise ValueError(f"{template_name} must be inside prediction_dir")
    return config


def paper_context(
    config: PipelineConfig,
    patch: PatchConfig,
    method: MethodConfig,
) -> dict[str, Any]:
    """Return the lightweight context recorded with canonical predictions."""

    return {
        "schema_version": 1,
        "config": str(config.source),
        "method": method.name,
        "patch_id": patch.id,
        "domain": patch.domain,
        "split": patch.split,
        "threshold_selection_metric": method.threshold_selection_metric,
    }


__all__ = [
    "BASELINE_METHOD_NAMES",
    "DEFAULT_CONFIG_PATH",
    "DOMAINS",
    "EXPECTED_PATCH_IDS",
    "EXPECTED_PATCH_RECORDS",
    "FIXED_THRESHOLDS",
    "LEARNED_METHOD_NAMES",
    "METHOD_NAMES",
    "MethodConfig",
    "PatchConfig",
    "PipelineConfig",
    "REQUIRED_ENVIRONMENT_VARIABLES",
    "SPLITS",
    "load_pipeline_config",
    "paper_context",
]
