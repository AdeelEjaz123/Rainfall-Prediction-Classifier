# Dataset: Australian Weather Observations

## Source & Citation
- **Primary Source:** Commonwealth of Australia, [Bureau of Meteorology (BOM)](http://www.bom.gov.au/climate/dwo/)
- **Distribution:** IBM Developer Skills Network / Cloud Object Storage
- **Direct Download URL:** [weatherAUS-2.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv)

## Overview & Scope
This project focuses on meteorological data collected across southeastern Australia, specifically filtered to the greater Melbourne metropolitan region:
- `Melbourne`
- `MelbourneAirport`
- `Watsonia`

After dropping records with missing features and restricting observations to the Melbourne geographic cluster, the dataset contains **7,557 daily records** across **23 variables**.

## Target Variable
- **`RainToday`** (derived from original BOM `RainTomorrow` column):
  - Binary classification target: `'Yes'` (rain recorded $\ge 1.0\text{ mm}$) or `'No'` ($< 1.0\text{ mm}$).
  - Class distribution in Melbourne subset:
    - **No:** 5,766 observations (**76.3%**)
    - **Yes:** 1,791 observations (**23.7%**)
  - *Class Imbalance:* ~3.2:1 ratio of dry to rainy days.

## Data Dictionary

| Variable | Type | Description | Unit |
| :--- | :--- | :--- | :--- |
| `Location` | Categorical | Weather station location | Nominal |
| `MinTemp` | Float | Minimum temperature in 24 hours to 9am | °C |
| `MaxTemp` | Float | Maximum temperature in 24 hours to 9am | °C |
| `Rainfall` | Float | Precipitation recorded in 24 hours to 9am | mm |
| `Evaporation` | Float | Class A pan evaporation in 24 hours to 9am | mm |
| `Sunshine` | Float | Bright sunshine hours in 24 hours to midnight | Hours |
| `WindGustDir` | Categorical | Direction of strongest wind gust in 24 hours | 16 compass points |
| `WindGustSpeed` | Float | Speed of strongest wind gust in 24 hours | km/h |
| `WindDir9am` | Categorical | Direction of wind at 9am | 16 compass points |
| `WindDir3pm` | Categorical | Direction of wind at 3pm | 16 compass points |
| `WindSpeed9am` | Float | Wind speed averaged over 10 minutes at 9am | km/h |
| `WindSpeed3pm` | Float | Wind speed averaged over 10 minutes at 3pm | km/h |
| `Humidity9am` | Float | Relative humidity at 9am | % |
| `Humidity3pm` | Float | Relative humidity at 3pm | % |
| `Pressure9am` | Float | Atmospheric pressure reduced to mean sea level at 9am | hPa |
| `Pressure3pm` | Float | Atmospheric pressure reduced to mean sea level at 3pm | hPa |
| `Cloud9am` | Float | Cloud cover at 9am | Oktas (0-8) |
| `Cloud3pm` | Float | Cloud cover at 3pm | Oktas (0-8) |
| `Temp9am` | Float | Temperature at 9am | °C |
| `Temp3pm` | Float | Temperature at 3pm | °C |
| `RainYesterday` | Categorical | Did it rain on the previous calendar day? | `'Yes'` / `'No'` |
| `Season` | Categorical | Australian meteorological season (derived) | Summer, Autumn, Winter, Spring |
| `RainToday` | Categorical | **Target:** Did it rain on the observation day? | `'Yes'` / `'No'` |
