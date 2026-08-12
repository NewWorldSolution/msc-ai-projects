"""Pydantic data contracts for EcoHome geocoding and weather data.

This module describes the shape of validated location and forecast data.
API calls and transformation logic belong in ``tools.py``.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class GeocodedLocation(BaseModel):
    """A location resolved by the geocoding service."""

    # api: ConfigDict(extra="forbid") rejects fields outside this model's contract.
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )
    timezone: str = Field(
        min_length=1,
        description="IANA timezone name",
    )

    # api: field_validator runs this check whenever Pydantic builds the model.
    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Reject timezone names that are absent from the IANA database."""

        try:
            # api: ZoneInfo loads an IANA timezone by its canonical key.
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc

        return value


class CurrentWeather(BaseModel):
    """Current conditions returned to the Energy Advisor."""

    temperature_c: float
    condition: str = Field(
        min_length=1,
        description="Normalized weather category used by the Energy Advisor",
    )
    humidity: float = Field(
        ge=0.0,
        le=100.0,
        description="Humidity as a percentage",
    )
    wind_speed: float = Field(
        ge=0.0,
        description="Wind speed in m/s",
    )


class HourlyWeather(BaseModel):
    """Weather and solar conditions for one forecast hour."""

    time: datetime

    temperature_c: float = Field(
        description="Temperature in degrees Celsius",
    )
    condition: str = Field(
        min_length=1,
        description="Normalized weather category used by the Energy Advisor",
    )
    solar_irradiance: float = Field(
        ge=0.0,
        description="Solar irradiance in W/m²",
    )
    humidity: float = Field(
        ge=0.0,
        le=100.0,
        description="Humidity as a percentage",
    )
    wind_speed: float = Field(
        ge=0.0,
        description="Wind speed in m/s",
    )
    is_schedulable: bool = Field(
        description="Whether this forecast hour is still available for scheduling",
    )

    @computed_field
    @property
    def hour(self) -> int:
        """Derive the local clock hour from the forecast timestamp."""

        return self.time.hour


class WeatherForecast(BaseModel):
    """Complete weather forecast returned by the weather tool."""

    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1)
    resolved_location: GeocodedLocation
    forecast_days: int = Field(ge=1, le=7)
    current: CurrentWeather
    hourly: list[HourlyWeather] = Field(min_length=1)
