"""XObjPrototype - Base model abstraction with validation and metadata support."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import Console

console = Console()

T = TypeVar("T", bound="XObjPrototype")


class XObjPrototype(BaseModel):
    """Base model abstraction with metadata and fuzzy search support."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={Decimal: str},
        arbitrary_types_allowed=True,
    )

    _metadata: dict[str, Any] = Field(default_factory=dict, alias="_metadata")
    _ns: str | None = Field(default=None, alias="_ns")
    _fuzzy_fields: ClassVar[list[str]] = []

    @field_validator("*", mode="before")
    @classmethod
    def coerce_types(cls, v: Any, info: Any) -> Any:
        """Coerce types before validation."""
        if info.field_name == "_metadata":
            return v if isinstance(v, dict) else {}
        return v

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update multiple metadata values."""
        self._metadata.update(metadata)

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self._metadata.clear()

    @property
    def ns(self) -> str:
        """Get the model namespace."""
        if self._ns:
            return self._ns
        return self.__class__.__name__.lower()

    @ns.setter
    def ns(self, value: str) -> None:
        """Set the model namespace."""
        self._ns = value

    def get_fuzzy_text(self) -> str:
        """Get concatenated text from fuzzy searchable fields."""
        fuzzy_parts = []
        for field in self._fuzzy_fields:
            if hasattr(self, field):
                value = getattr(self, field)
                if value is not None:
                    fuzzy_parts.append(str(value))
        return " ".join(fuzzy_parts)

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        """Convert model to dictionary."""
        data = self.model_dump(exclude={"_metadata", "_ns"})
        if include_metadata and self._metadata:
            data["_metadata"] = self._metadata
        return data

    def to_json(self, include_metadata: bool = False) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json(
            exclude={"_metadata", "_ns"} if not include_metadata else set()
        )

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        metadata = data.pop("_metadata", {})
        ns = data.pop("_ns", None)
        instance = cls(**data)
        if metadata:
            instance._metadata = metadata
        if ns:
            instance._ns = ns
        return instance

    def validate_fields(self) -> dict[str, Any]:
        """Validate all fields and return validation results."""
        results = {}
        for field_name, field_info in self.model_fields.items():
            try:
                value = getattr(self, field_name)
                self.model_validate({field_name: value})
                results[field_name] = {"valid": True, "value": value}
            except Exception as e:
                results[field_name] = {"valid": False, "error": str(e)}
        return results

    def clone(self: T, **kwargs: Any) -> T:
        """Create a clone of the model with optional field updates."""
        data = self.model_dump()
        data.update(kwargs)
        return self.__class__(**data)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.model_dump()})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(ns='{self.ns}', fields={list(self.model_fields.keys())})"


class UserModel(XObjPrototype):
    """Example user model extending XObjPrototype."""

    _fuzzy_fields: ClassVar[list[str]] = ["name", "email"]

    id: str
    name: str
    email: str
    age: int = Field(ge=0, le=150)
    role: str | None = Field(default="user")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()


class ProductModel(XObjPrototype):
    """Example product model extending XObjPrototype."""

    _fuzzy_fields: ClassVar[list[str]] = ["name", "sku", "category"]

    id: str
    name: str
    sku: str
    price: Decimal = Field(ge=0)
    category: str
    active: bool = True


class ConfigModel(XObjPrototype):
    """Example configuration model extending XObjPrototype."""

    _fuzzy_fields: ClassVar[list[str]] = ["key", "environment"]

    id: str
    key: str
    value: str | int | float | bool | dict | list
    environment: str = Field(default="dev")
    enabled: bool = True


class OrderModel(XObjPrototype):
    """Example order model extending XObjPrototype."""

    _fuzzy_fields: ClassVar[list[str]] = ["customer_name", "customer_email"]

    id: str
    user_id: str
    customer_name: str
    customer_email: str
    total: Decimal = Field(ge=0)
    status: str = Field(default="pending")
    items: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("customer_email")
    @classmethod
    def validate_customer_email(cls, v: str) -> str:
        """Validate customer email format."""
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate order status."""
        valid_statuses = {"pending", "processing", "shipped", "delivered", "cancelled"}
        if v not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        return v