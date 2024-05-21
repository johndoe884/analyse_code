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

    with open("plot.png", "wb") as f:
        f.write(buf.getbuffer())
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{img_str}"


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

    analysis_results["data"] = data
    analysis_results["var_results"] = var_results
    analysis_results["profit_loss"] = profit_loss
    analysis_results["total_profit_loss"] = total_profit_loss
    analysis_results["chart_url"] = chart_url

    return {"message": "Analysis completed successfully."}
