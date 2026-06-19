"""Weather module -- current conditions + a short forecast, no API key needed.

Uses Open-Meteo (free, keyless): geocode a place name, then fetch current
weather and today's high/low. Exposed as a `get_weather` tool the brain calls
when someone asks about the weather. A default location can be set in config so
"what's the weather?" (no place) still works.
"""
from __future__ import annotations

import logging
from typing import List

from ...core.module import Action, Module

log = logging.getLogger("everything_agent.modules.weather")

_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> short spoken phrases.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
    81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
    95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a severe thunderstorm with hail",
}


class WeatherModule(Module):
    name = "weather"

    def setup(self, robot, providers, memory, config) -> None:
        wcfg = (config or {}).get("weather", {}) or {}
        self.default_location = wcfg.get("default_location") or ""
        self.units = wcfg.get("units", "celsius")   # "celsius" | "fahrenheit"

    def actions(self) -> List[Action]:
        async def get_weather(args):
            place = (args.get("location") or self.default_location or "").strip()
            if not place:
                return ("I don't have a location set yet -- tell me a city and "
                        "I'll check, or set a default in my settings.")
            return await self._weather(place)

        return [Action(
            "get_weather",
            "Get the current weather and today's forecast for a place. "
            "Pass 'location' (a city/town name); omit to use the default location.",
            get_weather,
            params={"location": str},
        )]

    async def _weather(self, place: str) -> str:
        import httpx

        unit = "fahrenheit" if str(self.units).lower().startswith("f") else "celsius"
        sym = "F" if unit == "fahrenheit" else "C"
        try:
            async with httpx.AsyncClient(timeout=8.0) as http:
                g = await http.get(_GEOCODE, params={"name": place, "count": 1})
                results = (g.json() or {}).get("results") or []
                if not results:
                    return f"I couldn't find a place called {place}."
                loc = results[0]
                name = loc.get("name", place)
                country = loc.get("country", "")
                f = await http.get(_FORECAST, params={
                    "latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": unit, "wind_speed_unit": "kmh",
                    "timezone": "auto", "forecast_days": 1,
                })
                data = f.json() or {}
        except Exception as e:  # noqa: BLE001
            log.warning("weather lookup failed: %s", e)
            return f"I tried to check the weather in {place} but couldn't reach the service just now."

        cur = data.get("current", {}) or {}
        daily = data.get("daily", {}) or {}
        temp = round(cur.get("temperature_2m", 0))
        desc = _WMO.get(int(cur.get("weather_code", -1)), "unclear skies")
        wind = round(cur.get("wind_speed_10m", 0))
        where = f"{name}{', ' + country if country else ''}"
        out = f"In {where} it's {temp}°{sym} with {desc}, wind around {wind} km/h."
        try:
            hi = round(daily["temperature_2m_max"][0])
            lo = round(daily["temperature_2m_min"][0])
            out += f" Today's high is {hi}° and low {lo}°."
        except Exception:  # noqa: BLE001
            pass
        return out
