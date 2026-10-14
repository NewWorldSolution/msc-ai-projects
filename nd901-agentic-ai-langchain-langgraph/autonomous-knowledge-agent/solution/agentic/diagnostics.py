"""Failure descriptions that name schema-owned fields only, never model-chosen text."""

from pydantic import BaseModel, ValidationError


def schema_field_names(model: type[BaseModel]) -> set[str]:
    """Collect every property name the schema owns, including nested definitions."""
    # api: model_json_schema() returns the JSON Schema; nested models sit under $defs.
    schema = model.model_json_schema()
    names = set(schema.get("properties", {}))
    for definition in schema.get("$defs", {}).values():
        names.update(definition.get("properties", {}))
    return names


def schema_mismatch_detail(exc: ValidationError, model: type[BaseModel]) -> str:
    """Describe a validation failure by field path and error type.

    A location part that is not a schema field or a list index was chosen by the
    model (an extra key or a dictionary key), so it is replaced by a fixed label.
    """
    allowed = schema_field_names(model)
    parts = []
    # api: ValidationError.errors() lists each failed field with its location tuple and type code.
    for error in exc.errors()[:5]:
        # why: pydantic gives list indices as int; a numeric *string* is a model-chosen key.
        location = [str(part) if isinstance(part, int) or str(part) in allowed else "<unexpected-field>"
                    for part in error["loc"]]
        parts.append(".".join(location) + f" ({error['type']})")
    return "Schema mismatch: " + ", ".join(parts)
