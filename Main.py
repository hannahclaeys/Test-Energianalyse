#%%

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import cvxpy as cp
from collections import defaultdict

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.power as smp


#%%

# Task 1

# Load CSV files into pandas DataFrames
ElPrice: pd.DataFrame = pd.read_csv("PricesBuySell.csv")
ProsumerData: pd.DataFrame = pd.read_csv("ProsumerData.csv")

# Color palette for plots (per year + profit)
plotColors = {"2021": "#2E8B57",
             "2022": "#2BB673",
             "2023": "#20B2AA",
             "2024": "#1E90FF",
             "2025": "#0B3D91",
             "profit": 'xkcd:baby poop green'}
# pd.set_option('display.max_rows', None)
# print(ElPrice)
# print(ProsumerData)

# Apply identical time-feature engineering to both datasets
dataframes: list[pd.DataFrame] = [ProsumerData, ElPrice]

for df in dataframes:
    # Parse UTC timestamps and create DK local time (CET/CEST via tz_convert)
    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"], utc=True)
    df["TimeDK"] = df["TimeUTC"].dt.tz_convert("CET")

    # Extract calendar features for grouping/plotting
    df["Month"] = df["TimeDK"].dt.month
    df["Year"] = df["TimeDK"].dt.year
    df["DayOfMonth"] = df["TimeDK"].dt.day
    df["Hour"] = df["TimeDK"].dt.hour
    df["DayOfYear"] = df["TimeDK"].dt.dayofyear


#%%
# Yearly mean prices (buy/sell and non-transfer component)
ElPrice_mean_year: pd.DataFrame = ElPrice.groupby("Year").agg({"l_b": 'mean', "l_s": 'mean', "l_b_NT": 'mean'}).reset_index()
print(ElPrice_mean_year)

# Colors for the bar plot (one color per year)
colorlist: dict = [plotColors[str(year)] for year in ElPrice_mean_year["Year"]]

# Plot average sell price per year
plt.bar(ElPrice_mean_year["Year"].to_numpy(), ElPrice_mean_year["l_s"].to_numpy(), color=colorlist)
plt.xticks(list(range(2021, 2026)), rotation=45)
plt.grid(True, alpha=0.5)
plt.ylabel(r"Price [$DKK$]")
plt.title("Average daily electricity price for each year")
plt.savefig("Average daily electricity price for each year")
plt.show()

# Hourly mean sell price per year (average day shape)
ElPrice_mean_hourly_year: pd.DataFrame = ElPrice.groupby(["Year", "Hour"]).agg({"l_s": 'mean'}).reset_index()
ElPrice_mean_hourly_year = ElPrice_mean_hourly_year.pivot(index="Hour", columns="Year", values="l_s").reset_index()

print(ElPrice_mean_hourly_year)
print(ElPrice_mean_hourly_year.columns.values.tolist())
# years = ElPrice_mean_hourly_year.columns.drop("Hour")

# Plot hourly profiles for each year
hours = list(range(0, 24))
for columns in ElPrice_mean_hourly_year[[2021, 2022, 2023, 2024, 2025]].columns:
    li = ElPrice_mean_hourly_year[columns].to_list()
    plt.plot(hours, li, drawstyle="steps-post", label=str(columns), color=plotColors[f"{columns}"])
plt.grid(True, alpha=0.5)
plt.legend()
plt.xticks(list(range(24)))
plt.xlim(0, 23)
plt.ylabel(r"Sell price [$DKK$]")
plt.xlabel("Hours")
plt.title("Sell price for the average day at a given hour")
plt.savefig("Sell price for the average day at a given hour")
plt.show()

#%%

# Task 2

# Extract first 24 hours of sell prices (index slice includes an extra row, kept unchanged)
ElPrice_day1 = ElPrice.iloc[:25, ElPrice.columns.get_loc("l_s")].to_numpy()
print("Price the first 24 hours")
# print(ElPrice_day1)

# Battery parameters for optimization
params = {
    "Pmax": 5,
    "Cmax": 10,
    "Cmin": 1,
    "C_0": 5,
    "n_c": 0.95,
    "n_d": 0.95
}

from Task2 import DayliOptimizer, regression

# Loop setup (year/month/day traversal)
year = 2021
month = 1  # Jan
dayInMonth = 31  # Number of days in January

numberOfDayForAllMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Pre-allocate per-year arrays (365 assumed)
profit_list = np.zeros(365)
maxSellPrice = np.zeros(365)
minSellPrice = np.zeros(365)

# Store yearly arrays keyed by year
maxSPCombined = defaultdict(np.array)
minSPCombined = defaultdict(np.array)
profitEachYear = defaultdict(np.array)

# Store all daily charge/discharge/SOC trajectories (concatenated)
charging_array = np.empty(1)
discharging_array = np.empty(1)
SOC_array = np.empty(1)

DayNryear = 0  # Day index within the current year

# Iterate through all days from 2021 up to (but not including) 2026
while year < 2026:
    for day in range(1, dayInMonth + 1):

        # Build start/end timestamps for the day (DK timezone)
        t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0)).tz_localize("Europe/Copenhagen")
        t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0)).tz_localize("Europe/Copenhagen")

        # Extract 24 hourly sell prices for that day
        ElPrice_day = ElPrice.loc[(ElPrice["TimeDK"] >= t_s) & (ElPrice["TimeDK"] <= t_e), "l_s"].to_numpy()

        # Run daily battery arbitrage optimization
        profit, charging, discharging, SOC = DayliOptimizer(ElPrice_day, params)

        # Save daily max/min prices and profit
        ElPrice_sort = np.sort(ElPrice_day)
        maxSellPrice[DayNryear] = ElPrice_sort[-1]
        minSellPrice[DayNryear] = ElPrice_sort[0]
        profit_list[DayNryear] = profit

        # Concatenate trajectories for later interactive plotting
        charging_array = np.concatenate((charging_array, charging), axis=0)
        discharging_array = np.concatenate((discharging_array, discharging), axis=0)
        SOC_array = np.concatenate((SOC_array, SOC), axis=0)

        DayNryear += 1

    # Advance month/year bookkeeping
    month += 1
    if month > 12:
        # Store arrays for the finished year
        profitEachYear[f"{year}"] = profit_list
        maxSPCombined[f"{year}"] = maxSellPrice
        minSPCombined[f"{year}"] = minSellPrice

        # Reset arrays for next year
        profit_list = np.zeros(365)
        maxSellPrice = np.zeros(365)
        minSellPrice = np.zeros(365)

        print(f"The number of days in {year} is {DayNryear}")
        DayNryear = 0
        month = 1
        year += 1

    # Update days-in-month (simple non-leap-year calendar, kept unchanged)
    dayInMonth = numberOfDayForAllMonth[month - 1]
    # print(f"{dayInMonth=}")

#%%
def regression(xdata, ydata):
    # Simple linear regression using normal equations: y = beta0 + beta1*x
    ones = np.ones(len(xdata))
    z = np.vstack((ones, xdata))
    test = np.linalg.inv(np.matmul(z, z.T))
    test2 = np.matmul(z, ydata)
    answer = np.matmul(test, test2)
    return answer

# Combine all years into single arrays (inside loop kept unchanged)
for key, values in profitEachYear.items():
    dailyMaxSP = np.concatenate(list(maxSPCombined.values()))
    dailyMinSP = np.concatenate(list(minSPCombined.values()))
    dailyProfit = np.concatenate(list(profitEachYear.values()))

# Price spread per day across all years
dailyDiff = dailyMaxSP - dailyMinSP

# Fit linear model between price spread and profit
answer = regression(dailyDiff, dailyProfit)
print()

# Create regression line for plotting
maxValue = np.max(dailyDiff)
minValue = np.min(dailyDiff)
xValues = np.linspace(maxValue, minValue, 500)
yValues = answer[1] * xValues + answer[0]

plt.plot(xValues, yValues, "--k")
plt.show()

#%%

# Bar plot: average daily profit per year
plt.bar(range(len(profitEachYear)), [np.mean(value) for value in profitEachYear.values()], align="center", color=colorlist)
plt.ylabel("Profit [DKK]")
plt.title("Daily profit for the average for each year")
plt.xticks(range(len(profitEachYear)), list(profitEachYear.keys()))
plt.grid(True, alpha=0.5)
plt.savefig("Daily profit for the average for each year")
plt.show()

# Scatter: profit vs daily max sell price
for key in profitEachYear.keys():
    plt.scatter(maxSPCombined[key], profitEachYear[key], color=plotColors[key], label=key, alpha=0.8)
plt.xlabel("Max sell price of the day")
plt.ylabel("Profit for the day")
plt.legend()
plt.title("Daily profit over max sell price for the day")
plt.savefig("Daily profit over max sell price for the day")
plt.show()

# Scatter: profit vs daily min sell price
for key in profitEachYear.keys():
    plt.scatter(minSPCombined[key], profitEachYear[key], color=plotColors[key], label=key, alpha=0.8)
plt.xlabel("Minimum sell price for the day")
plt.ylabel("Profit for the day")
plt.title("Daily profit over minimum sell price for the day")
plt.legend()
plt.savefig("Daily profit over minimum sell price for the day")
plt.show()

# Scatter: profit vs daily price spread (max-min)
listDiff: list[np.array] = []
for key in profitEachYear.keys():
    listDiff.append(maxSPCombined[key] - minSPCombined[key])
    plt.scatter(maxSPCombined[key] - minSPCombined[key], profitEachYear[key], color=plotColors[key], label=key, alpha=0.8)
plt.xlabel("Diff sell price for the day")
plt.ylabel("Profit for the day")
plt.legend()
plt.title("Daily profit over diff in sell price for the day")
plt.savefig("Daily profit over diff in sell price for the day")
plt.show()

# Scatter: yearly subsets + overall regression line
for key in profitEachYear.keys():
    dailyDiffYearly = maxSPCombined[key] - minSPCombined[key]
    plt.scatter(dailyDiffYearly, profitEachYear[key], color=plotColors[key], label=key, alpha=0.8)
    print(f"The real average daily profit for that year was {np.mean(profitEachYear[key]):.2f}")

plt.xlabel("Diff sell price for the day")
plt.ylabel("Profit for the day")
plt.plot(xValues, yValues, "--", color="#8E9616")
plt.legend()
plt.title("Linear regression between daily profit and diff in sell price")
plt.savefig("Linear regression between daily profit and diff in sell price")
plt.show()

plt.close()
print(ElPrice_mean_year)
print(ElPrice_mean_year["l_s"].to_list())

# Expected profit per year using regression model and average spread per year
expectedProfit: list = [answer[0] + answer[1] * np.mean(value) for value in listDiff]
print(expectedProfit)

plt.bar(list(profitEachYear.keys()), expectedProfit, color=colorlist)
plt.grid(True, alpha=0.5)
plt.ylabel("Expected profit")
plt.title("Expected daily profit for each year")
plt.savefig("Expected daily profit for each year")
plt.show()

# fit = smf.ols(dailyDiff, profitEachYear).fit()
# print(fit.summary())


#%%
seePlot = True
dayNumber = 0
showMultiple = False

print("Type a number (int) and get a plot for the SOC of that day")
print("Print: Stop, to break the loop, use + to get the next day and minus for the previous day")

while seePlot:
    answer = input(">")
    plt.close()

    # Toggle multi-year mode
    if answer == "A":
        showMultiple = not showMultiple
        print(f"You have switched to {showMultiple=}")

    # Exit
    elif answer == "Stop":
        print("Stop")
        break

    # Step forward/backward one day
    elif answer == "+":
        print("In +")
        dayNumber += 1
    elif answer == "-":
        print("In -")
        dayNumber += -1

    # Jump to a specific day number
    elif answer.isnumeric():
        print("number")
        dayNumber = int(answer)
        dayNumber -= 1

    # Invalid input
    else:
        print("The input was not valid, only positive int or (+, -, Stop, A) is valid")
        print("If you want to end, type 'Stop'")
        continue

    print(f"{dayNumber+1=}")

    # Bounds check (5 years * 365 days = 1825)
    if dayNumber < 0 or dayNumber > 1825:
        print("There are no days in this range")
        print(f"{dayNumber-1}")
        print("dayNumber is set to 1")
        dayNumber = 1

    # Plot the same day-of-year across all years
    if dayNumber <= 365 and showMultiple:
        years = 2021
        for i in range(len(profitEachYear)):
            ElPrice_dayN = ElPrice.iloc[
                24 * (dayNumber) + (365 * i):24 * (dayNumber + 1) + (365 * i),
                ElPrice.columns.get_loc("l_s")
            ].to_numpy()

            profit, charging, discharging, SOC = DayliOptimizer(ElPrice_dayN, params)

            plt.plot(SOC, label=f"{years+i}", color=colorlist, alpha=0.8)
            print(f"{years + i}:{profit=:.2f} DKK", end=", ")
            print(f"Energy bought: {charging.sum():.2f} kWh", end=", ")
            print(f"Energy sold: {discharging.sum():.2f} kWh")

        plt.title(f"SOC for day {dayNumber+1} in each year")
        plt.ylim(0.5, 11.5)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=5, fancybox=True, shadow=True)
        plt.xticks(list(range(0, 24)))

    # Plot a single day (one year slice)
    else:
        ElPrice_dayN = ElPrice.iloc[24 * (dayNumber):24 * (dayNumber + 1), ElPrice.columns.get_loc("l_s")]
        print(ElPrice_dayN)

        ElPrice_dayN = ElPrice.iloc[24 * (dayNumber):24 * (dayNumber + 1), ElPrice.columns.get_loc("l_s")].to_numpy()
        print(len(ElPrice_dayN))

        profit, charging, discharging, SOC = DayliOptimizer(ElPrice_dayN, params)

        print(f"{profit=:.2f} DKK")
        print(len(SOC))
        print(charging)
        print(f"Energy bought {charging.sum()=:.2f} kWh")
        print(f"Energy sold {discharging.sum()=:.2f} kWh")

        # Dual-axis plot: price (left) and SOC (right)
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.step(list(range(24)), ElPrice_dayN, color=color)
        ax1.set_ylabel("Price", color=color)
        ax1.set_xlabel("Hours")
        ax1.set_xticks(list(range(0, 24)))
        ax1.set_xlim(0, 23)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_title(f"Day: {dayNumber}")

        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("SOC", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_yticks(list(range(1, 10)))
        ax2.plot(SOC, color=color)

    plt.grid(True)
    plt.show(block=False)
    # plt.pause(3)
    # plt.close()


# %%
