"""Regression tests for the weather model/tool integration boundary."""

import os
import unittest
from datetime import datetime as RealDatetime
from unittest.mock import patch

import tools as weather_tools
from models.weather import GeocodedLocation


class FixedDatetime(RealDatetime):
    """Provide a deterministic local clock while retaining datetime parsing."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 11, 12, 0, tzinfo=tz)


class FakeForecastResponse:
    """Minimal successful response implementing the requests methods we use."""

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "current": {
                "temperature_2m": 20.0,
                "relative_humidity_2m": 50.0,
                "weather_code": 0,
                "wind_speed_10m": 2.0,
            },
            "hourly": {
                "time": ["2026-08-11T11:00", "2026-08-11T13:00"],
                "temperature_2m": [19.0, 21.0],
                "relative_humidity_2m": [55.0, 45.0],
                "weather_code": [3, 0],
                "wind_speed_10m": [2.0, 2.5],
                "shortwave_radiation": [100.0, 500.0],
            },
        }


class WeatherToolTests(unittest.TestCase):
    """Protect the Pydantic construction and public error boundaries."""

    def test_weather_builder_supplies_schedulability(self):
        """Past and future API hours satisfy the HourlyWeather contract."""
        # api: patch.object temporarily replaces a dependency for one test.
        with (
            patch.dict(
                os.environ,
                {"FORECAST_URL": "https://weather.example.test"},
            ),
            patch.object(weather_tools, "datetime", FixedDatetime),
            patch.object(
                weather_tools,
                "geocode_location",
                lambda location: GeocodedLocation(
                    latitude=52.2297,
                    longitude=21.0122,
                    timezone="Europe/Warsaw",
                ),
            ),
            patch.object(
                weather_tools.requests,
                "get",
                lambda *args, **kwargs: FakeForecastResponse(),
            ),
        ):
            forecast = weather_tools._build_weather_forecast(
                "Warsaw, Poland",
                days=1,
            )

        self.assertEqual(
            [item.is_schedulable for item in forecast.hourly],
            [False, True],
        )
        self.assertTrue(all(item.time.tzinfo is not None for item in forecast.hourly))

    def test_public_weather_tool_returns_structured_error(self):
        """An internal failure becomes an agent-readable tool observation."""
        def fail_builder(location, days):
            raise ValueError("service unavailable")

        with patch.object(
            weather_tools,
            "_build_weather_forecast",
            fail_builder,
        ):
            result = weather_tools.get_weather_forecast.invoke(
                {"location": "Warsaw, Poland", "days": 1}
            )

        self.assertEqual(
            result,
            {"error": "Failed to get weather forecast: service unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
