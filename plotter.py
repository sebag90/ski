from datetime import date, datetime
import io
import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import meteostat as ms

# -----------------------------------------------------------------------------
# Core Plot Generation Function
# -----------------------------------------------------------------------------
def generate_plot(station_id, lat, lon, start_date, end_date, name=None, dpi=150):
    """
    Fetches Meteostat data and renders the 3-panel ski weather overview plot.
    Returns PNG image bytes.
    """
    pt = ms.Point(lat, lon)
    stations = ms.stations.nearby(pt, limit=5)

    if station_id:
        daily = ms.daily(station_id, start_date, end_date)
        df = ms.interpolate(daily, pt).fetch()
        # Backfill missing variables (e.g. wind gusts or sunshine) from nearby stations
        missing_cols = [
            col
            for col in ["temp", "tmin", "tmax", "prcp", "wspd", "wpgt", "tsun"]
            if col in df.columns and df[col].isna().all()
        ]
        if missing_cols:
            daily_nearby = ms.daily(stations, start_date, end_date)
            df_nearby = ms.interpolate(daily_nearby, pt).fetch()
            df = df.combine_first(df_nearby)
    else:
        daily = ms.daily(stations, start_date, end_date)
        df = ms.interpolate(daily, pt).fetch()

    if df.empty:
        raise ValueError(
            f"No weather data available for station {station_id} between {start_date} and {end_date}."
        )

    dates = df.index

    # Defensive extraction safe against missing/NAType values
    tmin_series = df["tmin"].dropna()
    tmax_series = df["tmax"].dropna()
    temp_series = df["temp"].dropna()
    prcp_series = df["prcp"].dropna()
    wpgt_series = df["wpgt"].dropna()
    wspd_series = df["wspd"].dropna()
    tsun_series = df["tsun"].dropna()

    avg_temp = float(temp_series.mean()) if not temp_series.empty else 0.0
    min_temp = float(tmin_series.min()) if not tmin_series.empty else -5.0
    max_temp = float(tmax_series.max()) if not tmax_series.empty else 10.0
    min_day = tmin_series.idxmin() if not tmin_series.empty else dates[0]

    total_prcp = float(prcp_series.sum()) if not prcp_series.empty else 0.0
    is_snow = (df["temp"].fillna(99) <= 0.5) & (df["prcp"].fillna(0) > 0)
    snow_days = int(is_snow.sum())
    snow_prcp_total = float(df.loc[is_snow, "prcp"].sum()) if snow_days > 0 else 0.0

    has_gust = not wpgt_series.empty
    max_gust = float(wpgt_series.max()) if has_gust else None

    has_sun = not tsun_series.empty
    sun_hours_total = float(tsun_series.sum()) / 60.0 if has_sun else None

    # Styling
    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        }
    )

    fig = plt.figure(figsize=(14, 11), facecolor="#f8fafc")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.95], hspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    start_str = start_date.strftime("%b %Y")
    end_str = end_date.strftime("%b %Y")
    if start_date.year == end_date.year and start_date.month == end_date.month:
        date_str = start_date.strftime("%B %Y")
    else:
        date_str = f"{start_str} – {end_str}"

    display_name = name or (
        f"Station {station_id}" if station_id else "Weather Overview"
    )
    fig.suptitle(
        f"{display_name} — Winter Weather & Ski Conditions ({date_str})",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
        y=0.975,
    )

    summary_parts = [
        f"Avg Temp: {avg_temp:+.1f}°C (Min: {min_temp:.1f}°C, Max: {max_temp:.1f}°C)",
        f"Precip: {total_prcp:.1f} mm (~{snow_prcp_total:.0f} cm fresh snow across {snow_days} snow days)",
    ]
    if sun_hours_total is not None:
        summary_parts.append(f"Sunshine: {sun_hours_total:.0f} hrs")
    if max_gust is not None:
        summary_parts.append(f"Max Gust: {max_gust:.0f} km/h")
    elif not wspd_series.empty:
        summary_parts.append(f"Max Wind: {float(wspd_series.max()):.0f} km/h")

    fig.text(
        0.5,
        0.948,
        "Summary:  " + "  •  ".join(summary_parts),
        ha="center",
        fontsize=9.5,
        color="#475569",
    )

    # -------------------------------------------------------------------------
    # Panel 1: Temperature
    # -------------------------------------------------------------------------
    ax1.set_facecolor("#ffffff")
    ymin = np.floor(min(min_temp - 3.5, -2))
    ymax = np.ceil(max(max_temp + 2.5, 4))
    ax1.set_ylim(ymin, ymax)

    ax1.axhline(
        0,
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
        label="0°C Freezing Line",
    )
    ax1.axhspan(
        ymin,
        0,
        facecolor="#eff6ff",
        alpha=0.65,
        zorder=1,
        label="Freezing Zone (Snow preservation)",
    )

    ax1.fill_between(
        dates,
        df["tmin"],
        df["tmax"],
        color="#bfdbfe",
        alpha=0.45,
        label="Daily Range (Min / Max)",
        zorder=2,
    )
    ax1.plot(
        dates,
        df["tmax"],
        color="#ef4444",
        linewidth=1.2,
        linestyle=":",
        alpha=0.85,
        label="Max Temp",
    )
    ax1.plot(
        dates,
        df["tmin"],
        color="#3b82f6",
        linewidth=1.2,
        linestyle=":",
        alpha=0.85,
        label="Min Temp",
    )
    ax1.plot(
        dates,
        df["temp"],
        color="#0f172a",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="Daily Mean Temp",
        zorder=4,
    )

    ax1.annotate(
        f"Coldest: {min_temp:.1f}°C",
        xy=(min_day, min_temp),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#1e40af",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#ffffff",
            edgecolor="#93c5fd",
            alpha=0.9,
        ),
        arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1),
    )

    ax1.set_ylabel(
        "Temperature (°C)", fontsize=10.5, fontweight="bold", color="#1e293b"
    )
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=6,
        frameon=False,
        fontsize=8.5,
    )
    ax1.grid(True)

    # -------------------------------------------------------------------------
    # Panel 2: Precipitation
    # -------------------------------------------------------------------------
    ax2.set_facecolor("#ffffff")
    snow_mask = df["temp"].fillna(99) <= 0.5
    snow_prcp = np.where(snow_mask, df["prcp"].fillna(0), 0)
    rain_prcp = np.where(~snow_mask, df["prcp"].fillna(0), 0)

    bar_w = 0.65
    ax2.bar(
        dates,
        snow_prcp,
        width=bar_w,
        color="#0284c7",
        label="Snow Precipitation (Mean Temp ≤ 0°C)",
        alpha=0.88,
        zorder=3,
    )
    ax2.bar(
        dates,
        rain_prcp,
        width=bar_w,
        color="#f59e0b",
        label="Rain / Wet Snow (Mean Temp > 0°C)",
        alpha=0.88,
        zorder=3,
    )

    for d, p, is_s in zip(dates, df["prcp"].fillna(0), snow_mask):
        if p >= 5.0:
            c = "#0369a1" if is_s else "#b45309"
            ax2.annotate(
                f"{p:.1f}",
                xy=(d, p),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=c,
            )

    p_max = float(prcp_series.max()) if not prcp_series.empty else 1.0
    ax2.set_ylabel(
        "Precipitation (mm)", fontsize=10.5, fontweight="bold", color="#1e293b"
    )
    ax2.set_ylim(0, max(p_max * 1.25, 10))
    ax2.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    ax2.grid(True)

    # -------------------------------------------------------------------------
    # Panel 3: Sunshine & Wind
    # -------------------------------------------------------------------------
    ax3.set_facecolor("#ffffff")
    if has_sun:
        sun_hours = df["tsun"].fillna(0) / 60.0
        ax3.bar(
            dates,
            sun_hours,
            width=bar_w,
            color="#fde047",
            edgecolor="#eab308",
            linewidth=0.8,
            alpha=0.6,
            label="Sunshine (hrs/day)",
            zorder=2,
        )
        ax3.set_ylabel(
            "Sunshine (hours)", fontsize=10.5, fontweight="bold", color="#a16207"
        )
        ax3.set_ylim(0, max(float(sun_hours.max()) * 1.25, 9))
    else:
        ax3.set_ylabel(
            "Sunshine (data N/A)", fontsize=10.5, fontweight="bold", color="#a16207"
        )
        ax3.set_ylim(0, 10)
    ax3.grid(True)

    ax3_wind = ax3.twinx()
    if has_gust:
        ax3_wind.plot(
            dates,
            df["wpgt"],
            color="#dc2626",
            linewidth=1.8,
            marker="^",
            markersize=3.5,
            label="Peak Gust (km/h)",
            zorder=4,
        )
    if not wspd_series.empty:
        ax3_wind.plot(
            dates,
            df["wspd"],
            color="#f87171",
            linewidth=1.1,
            linestyle="--",
            label="Avg Wind Speed (km/h)",
            zorder=3,
        )
    ax3_wind.axhline(
        60,
        color="#b91c1c",
        linestyle=":",
        linewidth=1.2,
        alpha=0.85,
        label="Lift Caution (60 km/h)",
    )
    ax3_wind.set_ylabel(
        "Wind Speed / Gust (km/h)", fontsize=10.5, fontweight="bold", color="#b91c1c"
    )

    w_top = max(
        max_gust or 0, float(wspd_series.max()) if not wspd_series.empty else 0, 50
    )
    ax3_wind.set_ylim(0, max(w_top * 1.2, 75))
    ax3_wind.grid(False)

    h3, l3 = ax3.get_legend_handles_labels()
    hw, lw = ax3_wind.get_legend_handles_labels()
    ax3.legend(
        h3 + hw,
        l3 + lw,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    # Date formatting
    day_span = len(df)
    interval = max(2, int(np.ceil(day_span / 16)))
    ax1.set_xlim(dates[0] - pd.Timedelta(days=0.7), dates[-1] + pd.Timedelta(days=0.7))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.subplots_adjust(top=0.915, bottom=0.065, left=0.065, right=0.935)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Standalone CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if "--save-only" in sys.argv or "--no-show" in sys.argv:
        matplotlib.use("Agg")
    else:
        matplotlib.use("webagg")

    LOCATION_NAME = "Embrun"
    POINT = ms.Point(44.5667, 6.5558)
    STATION_ID = "07591"  # Embrun
    START = date(2025, 12, 1)
    END = date(2026, 1, 31)

    img_data = generate_plot(
        station_id=STATION_ID,
        lat=POINT.latitude,
        lon=POINT.longitude,
        start_date=START,
        end_date=END,
        name=LOCATION_NAME,
        dpi=150,
    )

    output_file = "ski_weather.png"
    with open(output_file, "wb") as f:
        f.write(img_data)
    print(f"Saved plot image to {output_file}")

    if "--save-only" not in sys.argv and "--no-show" not in sys.argv:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(img_data))
            fig, ax = plt.subplots(figsize=(14, 11))
            ax.imshow(img)
            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.show()
        except Exception:
            pass
