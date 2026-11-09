Manual acquisition and local processing (route used for the reported results)

1. pNEUMA archive (Zenodo 10.5281/zenodo.10491409, CC BY 4.0), 15.76 GB, MD5 4e842b2200bea2f80d41c6b785a9de25
   https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip?download=1
2. Open-Meteo (browser download, CSV):
   ERA5 realised:   https://archive-api.open-meteo.com/v1/archive?latitude=37.9825&longitude=23.73&start_date=2024-01-01&end_date=2026-10-31&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m&models=era5&wind_speed_unit=ms&timezone=Europe%2FAthens&format=csv
   Day-ahead:       https://previous-runs-api.open-meteo.com/v1/forecast?latitude=37.9825&longitude=23.73&start_date=2024-01-01&end_date=2026-10-31&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_speed_10m_previous_day1,wind_speed_100m_previous_day1,wind_direction_100m_previous_day1,wind_speed_100m_previous_day2,wind_direction_100m_previous_day2&models=ecmwf_ifs025&wind_speed_unit=ms&timezone=Europe%2FAthens&format=csv
   Short-range:     https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=37.9825&longitude=23.73&start_date=2024-01-01&end_date=2026-10-31&hourly=wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m&models=ecmwf_ifs025&wind_speed_unit=ms&timezone=Europe%2FAthens&format=csv
3. OpenStreetMap buildings (Overpass API, JSON; saved as export.json):
   https://overpass-api.de/api/interpreter?data=%5Bout%3Ajson%5D%5Btimeout%3A300%5D%3B%28way%5B%22building%22%5D%2837.965%2C23.710%2C38.000%2C23.750%29%3Brelation%5B%22building%22%5D%2837.965%2C23.710%2C38.000%2C23.750%29%3B%29%3Bout%20geom%3B
4. Place acquire_and_process.py, overpass_to_geojson.py, run_parts.py, finalize.py and
   pneuma_file_list.csv in a folder next to the downloads (finalize.py expects the downloads in
   the parent folder), then:
     python3 run_parts.py ../pNEUMA_dataset.zip pneuma_file_list.csv parts 3600 20181024_d4_0830_0900.csv
     python3 finalize.py          # -> processed_inputs.zip
   finalize.py identifies the wind files by their columns: the day-ahead file by its previous_day
   columns, and of the other two the complete one (ERA5) and the one with gaps (historical forecast); check
   acquisition_log.jsonl to confirm the mapping.
