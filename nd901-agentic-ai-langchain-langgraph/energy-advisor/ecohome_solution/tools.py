"""
Tools for EcoHome Energy Advisor Agent
"""

import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pycountry
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from models.energy import DatabaseManager
from models.weather import (
    GeocodedLocation,
    CurrentWeather,
    HourlyWeather,
    WeatherForecast,
)

# api: load_dotenv() reads .env into os.environ. Needed here because this module
# is imported directly by the notebooks and must not depend on agent.py running first.
load_dotenv()

# Initialize database manager
db_manager = DatabaseManager()


def get_embeddings() -> OpenAIEmbeddings:
    """
    Build the embeddings client used for BOTH writing and reading the vector store.

    The store must be written and queried by the same model at the same endpoint.
    02_rag_setup.ipynb writes it; search_energy_tips below reads it. If the two
    disagree, similarity search returns noise rather than an error.
    """
    return OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("VOCAREUM_API_KEY"),
    )


def geocode_location(location: str) -> GeocodedLocation:
    """Convert a location name to latitude, longitude, and timezone.
    Args:
        location (str): Location name (e.g., "Warsaw, Poland"
                                             "San Francisco, California, USA"
                                             "00-001, Poland"
                                             "94103, USA")

    """
    geocoding_url = os.getenv("GEOCODING_URL")
    if not geocoding_url:
        raise ValueError("GEOCODING_URL not set in environment variables")
    if location is None or location.strip() == "":
        raise ValueError("Location must be a non-empty string")
    parts = [part.strip() for part in location.split(",")]
    if len(parts) not in (2, 3) or any(part == "" for part in parts):
        raise ValueError(
            "Location must be in the format 'City, Country' or 'City, State, Country' or 'Postal Code, Country'"
        )
    place = parts[0]
    region = parts[1] if len(parts) == 3 else None
    country = parts[-1]
    try:
        country_record = pycountry.countries.lookup(country)
    except LookupError as exc:
        raise ValueError(f"Unknown country: {country}") from exc
    params = {
        "name": place,
        "countryCode": country_record.alpha_2,
        "language": "en",
        "format": "json",
        "count": 10,
    }
    try:
        response = requests.get(geocoding_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise ValueError(f"Error during geocoding request: {exc}") from exc
    results = data.get("results", [])
    if region:
        results = [
            candidate
            for candidate in results
            if candidate.get("admin1", "").casefold() == region.casefold()
        ]
    if not results:
        raise ValueError(f"Could not geocode location: {location}")
    result = results[0]
    return GeocodedLocation(
        latitude=result["latitude"],
        longitude=result["longitude"],
        timezone=result["timezone"],
    )


def categorize_weather_code(code: int) -> str:
    """Convert an Open-Meteo WMO code into a database weather category.

    Args:
        code: Open-Meteo WMO weather code.

    Returns:
        One of: "sunny", "partly_cloudy", "cloudy", "rainy", or "unknown".
    """
    if code in (0, 1):
        return "sunny"

    if code == 2:
        return "partly_cloudy"

    if code in (3, 45, 48, 71, 73, 75, 77, 85, 86):
        return "cloudy"

    if code in (
        51,
        53,
        55,
        56,
        57,
        61,
        63,
        65,
        66,
        67,
        80,
        81,
        82,
        95,
        96,
        99,
    ):
        return "rainy"

    return "unknown"


def _build_weather_forecast(location: str, days: int = 3) -> WeatherForecast:
    """
        Retrieve a validated weather forecast for a human-readable location.

    The location is resolved internally into latitude, longitude, and an IANA
    timezone before the weather forecast is requested.

    Args:
        location (str): Location in one of the supported formats:
            "City, Country", "City, State, Country", or
            "Postal Code, Country".
        days (int): Number of forecast days to retrieve, from 1 to 7.

    Returns:
        WeatherForecast: A validated weather forecast containing:
            - The original and resolved location
            - The number of requested forecast days
            - Current temperature, condition, humidity, and wind speed
            - Hourly timestamps, local clock hours, temperature, condition,
              solar irradiance, humidity, and wind speed

            Temperatures are expressed in degrees Celsius, humidity as a
            percentage, wind speed in metres per second, and solar irradiance
            in watts per square metre.

    Raises:
        ValueError: If the location or number of days is invalid, the location
            cannot be resolved, an external request fails, or the returned data
            fails validation.
    """

    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 7:
        raise ValueError("days must be an integer between 1 and 7")

    forecast_url = os.getenv("FORECAST_URL")
    if not forecast_url:
        raise ValueError("FORECAST_URL not set in environment variables")

    resolved_location = geocode_location(location)
    local_timezone = ZoneInfo(resolved_location.timezone)
    local_now = datetime.now(local_timezone)
    params = {
        "latitude": resolved_location.latitude,
        "longitude": resolved_location.longitude,
        "timezone": resolved_location.timezone,
        "forecast_days": days,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "current": ("temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"),
        "hourly": (
            "temperature_2m,relative_humidity_2m,weather_code,"
            "wind_speed_10m,shortwave_radiation"
        ),
    }

    try:
        response = requests.get(forecast_url, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
    except requests.RequestException as exc:
        raise ValueError(f"Error during weather forecast request: {exc}") from exc

    current = weather_data["current"]
    hourly = weather_data["hourly"]
    current_weather = CurrentWeather(
        temperature_c=current["temperature_2m"],
        condition=categorize_weather_code(current["weather_code"]),
        humidity=current["relative_humidity_2m"],
        wind_speed=current["wind_speed_10m"],
    )
    hourly_forecast = []
    for index, forecast_time in enumerate(hourly["time"]):
        forecast_timestamp = datetime.fromisoformat(forecast_time)
        if forecast_timestamp.tzinfo is None:
            # api: replace(tzinfo=...) attaches the API's known local timezone
            # without changing the local clock reading.
            forecast_timestamp = forecast_timestamp.replace(tzinfo=local_timezone)
        else:
            # api: astimezone converts an aware timestamp to the location timezone.
            forecast_timestamp = forecast_timestamp.astimezone(local_timezone)

        hourly_forecast.append(
            HourlyWeather(
                time=forecast_timestamp,
                temperature_c=hourly["temperature_2m"][index],
                condition=categorize_weather_code(hourly["weather_code"][index]),
                solar_irradiance=hourly["shortwave_radiation"][index],
                humidity=hourly["relative_humidity_2m"][index],
                wind_speed=hourly["wind_speed_10m"][index],
                is_schedulable=forecast_timestamp >= local_now,
            )
        )

    forecast = WeatherForecast(
        location=location,
        resolved_location=resolved_location,
        forecast_days=days,
        current=current_weather,
        hourly=hourly_forecast,
    )

    return forecast


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """Retrieve hourly weather and solar conditions for a location.

    Args:
        location: A city and country, city/state/country, or postal code/country.
        days: Number of forecast days to retrieve, from 1 to 7.

    Returns:
        A JSON-compatible validated forecast. If retrieval or validation fails,
        returns a dictionary containing an ``error`` message.
    """
    try:
        forecast = _build_weather_forecast(location, days)
        # api: mode="json" converts nested models and datetimes to JSON-safe values.
        return forecast.model_dump(mode="json")
    except Exception as exc:
        # why: Agent-facing tools use one structured error contract so an
        # external-service failure does not terminate the entire graph run.
        return {"error": f"Failed to get weather forecast: {exc}"}


@tool
def get_electricity_prices(date: str = None, device_type: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today). This project
            uses a static time-of-use tariff, so valid dates share the same
            hourly schedule.
        device_type (str): Optional User-facing aliases such as "electric car", "thermostat",
            "dishwasher", and "pool pump" are mapped to the corresponding
            database device category. When omitted, generic household pricing
            is returned.

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates
        E.g:
        prices = {
            "date": ...,
            "device_type": ...,
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "hourly_rates": [
                {
                    "hour": .., # for hour in range(24)
                    "rate": ..,
                    "period": ..,
                    "demand_charge": ...
                }
            ]
        }

        For this project mock, ``rate`` is the all-in price per kWh.
        ``demand_charge`` shows the peak portion already included in that rate.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        # Validate date format
        datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return {"error": "Invalid date format. Expected YYYY-MM-DD"}

    base_rate = 0.10  # Original database off-peak rate in USD per kWh
    peak_demand_charge = 0.05  # Peak portion included in the all-in rate
    peak_hours_by_device = {
        "ev": ("EV", range(18, 22)),
        "hvac": ("HVAC", range(12, 18)),
        "appliance": ("appliance", range(19, 23)),
    }
    device_aliases = {
        "ev": "ev",
        "electric vehicle": "ev",
        "electric car": "ev",
        "car": "ev",
        "hvac": "hvac",
        "thermostat": "hvac",
        "air conditioner": "hvac",
        "heat pump": "hvac",
        "appliance": "appliance",
        "dishwasher": "appliance",
        "washing machine": "appliance",
        "dryer": "appliance",
        "pool pump": "appliance",
    }

    if device_type is None:
        peak_hours = range(6, 22)  # Default peak hours for general pricing
        resolved_device_type = "household"
    else:
        device_key = device_type.strip().casefold()
        canonical_key = device_aliases.get(device_key)

        if canonical_key is None:
            return {
                "error": (
                    f"Unknown device type: {device_type}. "
                    "Expected EV, HVAC, appliance, or a supported device alias."
                )
            }

        resolved_device_type, peak_hours = peak_hours_by_device[canonical_key]

    hourly_rates = []
    for hour in range(24):
        if hour in peak_hours:
            rate = base_rate + peak_demand_charge
            period = "peak"
            demand_charge = peak_demand_charge
        else:
            rate = base_rate
            period = "off_peak"
            demand_charge = 0.0

        # why: `rate` is already the all-in value used by the historical data;
        # consumers must not add `demand_charge` to it a second time.
        hourly_rates.append(
            {
                "hour": hour,
                "rate": round(rate, 4),
                "period": period,
                "demand_charge": round(demand_charge, 4),
            }
        )

    return {
        "date": date,
        "device_type": resolved_device_type,
        "pricing_type": "time_of_use",
        "currency": "USD",
        "unit": "per_kWh",
        "hourly_rates": hourly_rates,
    }


@tool
def query_energy_usage(
    start_date: str, end_date: str, device_type: str = None
) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)

        if device_type:
            records = [r for r in records if r.device_type == device_type]

        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(
                sum(record.consumption_kwh for record in records),
                2,
            ),
            "total_cost_usd": round(
                sum(record.cost_usd or 0.0 for record in records),
                2,
            ),
            "device_breakdown": {},
            "hourly_breakdown": {},
        }

        for record in records:
            device_key = record.device_type or "unknown"

            if device_key not in usage_data["device_breakdown"]:
                usage_data["device_breakdown"][device_key] = {
                    "consumption_kwh": 0.0,
                    "cost_usd": 0.0,
                    "records": 0,
                }

            device_summary = usage_data["device_breakdown"][device_key]
            device_summary["consumption_kwh"] += record.consumption_kwh
            device_summary["cost_usd"] += record.cost_usd or 0.0
            device_summary["records"] += 1

            hour_key = f"{record.timestamp.hour:02d}:00"

            if hour_key not in usage_data["hourly_breakdown"]:
                usage_data["hourly_breakdown"][hour_key] = {
                    "consumption_kwh": 0.0,
                    "cost_usd": 0.0,
                    "records": 0,
                }

            hourly_summary = usage_data["hourly_breakdown"][hour_key]
            hourly_summary["consumption_kwh"] += record.consumption_kwh
            hourly_summary["cost_usd"] += record.cost_usd or 0.0
            hourly_summary["records"] += 1

        for summary in usage_data["device_breakdown"].values():
            summary["consumption_kwh"] = round(summary["consumption_kwh"], 2)
            summary["cost_usd"] = round(summary["cost_usd"], 2)

        for summary in usage_data["hourly_breakdown"].values():
            summary["consumption_kwh"] = round(summary["consumption_kwh"], 2)
            summary["cost_usd"] = round(summary["cost_usd"], 2)

        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)

        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records)
                / max(1, (end_dt - start_dt).days),
                2,
            ),
            "records": [],
        }

        for record in records:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": record.generation_kwh,
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )

        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        weather_counts = Counter(
            record.weather_condition or "unknown" for record in generation_records
        )

        # api: most_common(1) returns the most frequent value and its count.
        dominant_weather = (
            weather_counts.most_common(1)[0][0] if weather_counts else "unknown"
        )
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(
                    sum(r.consumption_kwh for r in usage_records), 2
                ),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": dominant_weather,
            },
        }

        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += (
                record.consumption_kwh
            )
            summary["usage"]["device_breakdown"][device]["cost_usd"] += (
                record.cost_usd or 0
            )
            summary["usage"]["device_breakdown"][device]["records"] += 1

        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load every source article so the knowledge base can grow without
            # changing this tool.
            documents = []
            document_directory = Path("data/documents")
            # api: glob("*.txt") discovers all text documents in the directory.
            document_paths = sorted(document_directory.glob("*.txt"))
            if not document_paths:
                raise ValueError(
                    f"No knowledge documents found in {document_directory}"
                )

            for doc_path in document_paths:
                loader = TextLoader(str(doc_path))
                documents.extend(loader.load())

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            if not splits:
                raise ValueError("Knowledge documents produced no text chunks")

            # Create vector store
            embeddings = get_embeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
        else:
            # Load existing vector store
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            )

        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)

        results = {"query": query, "total_results": len(docs), "tips": []}

        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high"
                    if i < 2
                    else "medium"
                    if i < 4
                    else "low",
                }
            )

        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
    period_days: float = 1.0,
) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
        period_days (float): Number of days represented by the usage figures.
            Defaults to one day.

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    if period_days <= 0:
        return {"error": "period_days must be greater than 0"}

    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (
        (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    )
    annual_savings_usd = savings_usd * (365 / period_days)

    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "period_days": period_days,
        "annual_savings_usd": round(annual_savings_usd, 2),
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
