import math
import random
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pandas_datareader import data as pdr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

yf.pdr_override()

app = FastAPI()

# Data storage for analysis results
analysis_results = {
    "data": None,
    "var_results": None,
    "profit_loss": None,
    "total_profit_loss": None,
    "chart_url": None,
}


class AnalysisRequest(BaseModel):
    h: int
    d: int
    t: str
    p: int


def get_stock_data(symbol, start, end):
    try:
        data = pdr.get_data_yahoo(symbol, start=start, end=end)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching data for {symbol}: {e}"
        )


def add_signals(data, body_threshold=0.01):
    data["Buy"] = 0
    data["Sell"] = 0

    for i in range(2, len(data)):
        # Three Soldiers
        if (
            (data.Close[i] - data.Open[i]) >= body_threshold
            and data.Close[i] > data.Close[i - 1]
            and (data.Close[i - 1] - data.Open[i - 1]) >= body_threshold
            and data.Close[i - 1] > data.Close[i - 2]
            and (data.Close[i - 2] - data.Open[i - 2]) >= body_threshold
        ):
            data.at[data.index[i], "Buy"] = 1

        # Three Crows
        if (
            (data.Open[i] - data.Close[i]) >= body_threshold
            and data.Close[i] < data.Close[i - 1]
            and (data.Open[i - 1] - data.Close[i - 1]) >= body_threshold
            and data.Close[i - 1] < data.Close[i - 2]
            and (data.Open[i - 2] - data.Close[i - 2]) >= body_threshold
        ):
            data.at[data.index[i], "Sell"] = 1

    return data


def simulate_var(data, signal_type, min_history=101, shots=10000, period=7):
    results = []
    for i in range(min_history, len(data) - period):
        if (signal_type == "buy" and data.Buy[i] == 1) or (
            signal_type == "sell" and data.Sell[i] == 1
        ):
            mean = data.Close[i - min_history : i].pct_change(1).mean()
            std = data.Close[i - min_history : i].pct_change(1).std()
            simulated = [random.gauss(mean, std) for _ in range(shots)]
            simulated.sort(reverse=True)
            var95 = simulated[int(len(simulated) * 0.95)]
            var99 = simulated[int(len(simulated) * 0.99)]
            results.append((data.index[i], var95, var99))
    return results


def calculate_profit_loss(data, signal_type, period):
    profit_loss = []
    total_profit_loss = 0
    for i in range(1, len(data) - period):
        if signal_type == "buy" and data.Buy[i] == 1:
            profit = data.Close[i + period] - data.Close[i]
            profit_loss.append(profit)
            total_profit_loss += profit
        elif signal_type == "sell" and data.Sell[i] == 1:
            profit = data.Close[i] - data.Close[i + period]
            profit_loss.append(profit)
            total_profit_loss += profit
    return profit_loss, total_profit_loss


def plot_signals(data, var_results):
    dates = [result[0] for result in var_results]
    var95_values = [result[1] for result in var_results]
    var99_values = [result[2] for result in var_results]

    avg_var95 = sum(var95_values) / len(var95_values)
    avg_var99 = sum(var99_values) / len(var99_values)

    plt.figure(figsize=(14, 7))
    plt.plot(dates, var95_values, label="95% VaR")
    plt.plot(dates, var99_values, label="99% VaR")
    plt.axhline(y=avg_var95, color="r", linestyle="-", label="Average 95% VaR")
    plt.axhline(y=avg_var99, color="g", linestyle="-", label="Average 99% VaR")
    plt.title("VaR Analysis")
    plt.xlabel("Date")
    plt.ylabel("Value at Risk")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{img_str}"


def convert_to_serializable(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="list")
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data


@app.get("/status")
def status():
    return {"status": True}


@app.post("/analyse")
async def analyse(request: AnalysisRequest):
    symbol = "AMZN"
    today = date.today()
    time_past = today - timedelta(days=1095)

    data = get_stock_data(symbol, time_past, today)
    data = add_signals(data)
    var_results = simulate_var(
        data,
        signal_type=request.t,
        min_history=request.h,
        shots=request.d,
        period=request.p,
    )
    profit_loss, total_profit_loss = calculate_profit_loss(
        data, signal_type=request.t, period=request.p
    )
    chart_url = plot_signals(data, var_results)

    analysis_results["data"] = convert_to_serializable(data)
    analysis_results["var_results"] = convert_to_serializable(var_results)
    analysis_results["profit_loss"] = convert_to_serializable(profit_loss)
    analysis_results["total_profit_loss"] = convert_to_serializable(total_profit_loss)
    analysis_results["chart_url"] = chart_url

    return {"analysis_results": analysis_results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
