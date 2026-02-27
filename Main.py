
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

# Import csv files into pandas

ElPrice: pd.DataFrame = pd.read_csv("PricesBuySell.csv")
ProsumerData: pd.DataFrame = pd.read_csv("ProsumerData.csv")

plotColors = {"2021": "#2E8B57",
             "2022": "#2BB673",
             "2023": "#20B2AA",
             "2024": "#1E90FF",
             "2025": "#0B3D91",
             "profit": 'xkcd:baby poop green'}
#pd.set_option('display.max_rows', None)
#print(ElPrice)
#print(ProsumerData)

dataframes: list[pd.DataFrame] = [ProsumerData,ElPrice]

for df in dataframes:
    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"], utc=True)
    df["TimeDK"]  = df["TimeUTC"].dt.tz_convert("CET")
    df["Month"] = df["TimeDK"].dt.month
    df["Year"] = df["TimeDK"].dt.year
    df["DayOfMonth"] = df["TimeDK"].dt.day
    df["Hour"] = df["TimeDK"].dt.hour
    df["DayOfYear"] = df["TimeDK"].dt.dayofyear


#%%
ElPrice_mean_year: pd.DataFrame = ElPrice.groupby("Year").agg({"l_b":'mean','l_s':'mean','l_b_NT':'mean'}).reset_index()
print(ElPrice_mean_year)
colorlist: dict = [plotColors[str(year)] for  year in ElPrice_mean_year['Year']]

#ElPrice_mean_year.plot(kind="bar", x ="Year", y="l_b", title = "test", color = colorlist)
plt.bar(ElPrice_mean_year["Year"].to_numpy(), ElPrice_mean_year["l_s"].to_numpy(), color = colorlist)
plt.xticks(list(range(2021,2026)), rotation = 45)
plt.grid(True, alpha = 0.5)
plt.ylabel(r"El-price [$DKK$]")
plt.title("Avarage daily elektrisity price for each year")
plt.savefig("Avarage daily elektrisity price for each year")
plt.show()


ElPrice_mean_hourly_year: pd.DataFrame = ElPrice.groupby(['Year','Hour']).agg({"l_s":'mean'}).reset_index()
ElPrice_mean_hourly_year = ElPrice_mean_hourly_year.pivot(index="Hour", columns="Year", values = "l_s").reset_index()

#ElPrice_mean_hourly_year.plot(kind="bar", x ="Year", y="l_b", title = "test")
print(ElPrice_mean_hourly_year)
print(ElPrice_mean_hourly_year.columns.values.tolist())
#years = ElPrice_mean_hourly_year.columns.drop("Hour")


#ElPrice_mean_hourly_year.plot(kind="bar", x= "Hour", title = "test2")

#print(ElPrice_mean_hourly_year[2021])


hours = list(range(0,24))
for columns in ElPrice_mean_hourly_year[[2021,2022,2023,2024,2025]].columns:
    li = ElPrice_mean_hourly_year[columns].to_list()
    plt.plot(hours, li, drawstyle = "steps-post", label = str(columns), color = plotColors[f"{columns}"])
plt.grid(True, alpha = 0.5)
plt.legend()
plt.xticks(list(range(24)))
plt.xlim(0,23)
plt.ylabel(r"Sell price [$DKK$]")
plt.xlabel("Hours")
plt.title("Sell price for the avarage day at a given hour")
plt.savefig("Sell price for the avarage day at a given hour")
plt.show()

#%%

#Task 2x

ElPrice_day1 = ElPrice.iloc[:25,ElPrice.columns.get_loc('l_s')].to_numpy()
print("El pricen de første 24 timer")
#print(ElPrice_day1)

params = {
    "Pmax": 5,
    "Cmax": 10,
    "Cmin": 1,
    "C_0": 5,
    "n_c": 0.95,
    "n_d": 0.95
}

from Task2 import DayliOptimizer, regression

year = 2021
month = 1 # Jan
dayInMonth = 31 #Antal dage i Januar

numberOfDayForAllMonth = [31,28,31,30,31,30,31,31,30,31,30,31]
profit_list = np.zeros(365)
maxSellPrice = np.zeros(365)
minSellPrice = np.zeros(365)
maxSPCombined = defaultdict(np.array)
minSPCombined = defaultdict(np.array)
profitEachYear = defaultdict(np.array)
charging_array = np.empty(1)
discharging_array = np.empty(1)
SOC_array = np.empty(1)
DayNryear = 0 # Hvilken dag vi er noget til i året

while year < 2026:
    for day in range(1, dayInMonth+1):

        #print(f"{year}:{month}:{day}")
        t_s = pd.Timestamp(dt.datetime(year, month, day, 0, 0, 0)).tz_localize("Europe/Copenhagen")
        t_e = pd.Timestamp(dt.datetime(year, month, day, 23, 0, 0)).tz_localize("Europe/Copenhagen")
        ElPrice_day = ElPrice.loc[(ElPrice["TimeDK"] >= t_s) & (ElPrice["TimeDK"] <= t_e), "l_s"].to_numpy()
        profit, charging, discharging, SOC = DayliOptimizer(ElPrice_day, params)
        ElPrice_sort = np.sort(ElPrice_day)
        maxSellPrice[DayNryear] = ElPrice_sort[-1]
        minSellPrice[DayNryear] = ElPrice_sort[0]
        profit_list[DayNryear] = profit

        charging_array = np.concatenate((charging_array, charging), axis = 0)
        discharging_array = np.concatenate((discharging_array, discharging), axis = 0)
        SOC_array = np.concatenate((SOC_array, SOC), axis= 0)
        DayNryear += 1
    month += 1
    if month > 12:
        profitEachYear[f"{year}"] = profit_list
        maxSPCombined[f"{year}"] = maxSellPrice
        minSPCombined[f"{year}"] = minSellPrice
        profit_list = np.zeros(365)
        maxSellPrice = np.zeros(365)
        minSellPrice = np.zeros(365)
        print(f"The number of days in {year} is {DayNryear}")
        DayNryear = 0
        month = 1
        year += 1
    dayInMonth = numberOfDayForAllMonth[month-1]
    #print(f"{dayInMonth=}")

#%%
def regression(xdata, ydata):
    ones = np.ones(len(xdata))
    z= np.vstack((ones, xdata))
    test = np.linalg.inv(np.matmul(z, z.T))
    test2 = np.matmul(z, ydata)
    #print(test)
    #print(test.shape)
    answer = np.matmul(test, test2)
    #print(answer)
    #print(type(answer))
    return answer

for key, values in profitEachYear.items():
    dailyMaxSP = np.concatenate(list(maxSPCombined.values()))
    dailyMinSP = np.concatenate(list(minSPCombined.values()))
    dailyProfit = np.concatenate(list(profitEachYear.values()))


#print(dailyMaxSP)

dailyDiff = dailyMaxSP - dailyMinSP

answer = regression(dailyDiff, dailyProfit)
print()


#plt.plot(np.sort(dailyDiff), np.poly1d(answer,1)(np.sort(dailyDiff)), "--k")
maxValue = np.max(dailyDiff)
minValue = np.min(dailyDiff)
xValues = np.linspace(maxValue, minValue, 500)
yValues = answer[1] * xValues +  answer[0]


plt.plot(xValues, yValues, "--k")
plt.show()

#%%



plt.bar(range(len(profitEachYear)),[np.mean(value) for value in profitEachYear.values()],align = 'center', color = colorlist)
plt.ylabel("Profit [DKK]")
plt.title("Daily profit for the avarage for each year")
plt.xticks(range(len(profitEachYear)),list(profitEachYear.keys()))
plt.grid(True, alpha = 0.5)
plt.savefig("Daily profit for the avarage for each year")
plt.show()

for key in profitEachYear.keys():
    plt.scatter(maxSPCombined[key], profitEachYear[key], color = plotColors[key], label = key, alpha = 0.8)
#plt.scatter(dailyMaxSP, dailyProfit)
plt.xlabel("Max sell price for the day")
plt.ylabel("Profit for the day")
plt.legend()
plt.title("Daily profit over max sell price for the day")
plt.savefig("Daily profit over max sell price for the day")
plt.show()

#plt.scatter(dailyMinSP, dailyProfit)
for key in profitEachYear.keys():
    plt.scatter(minSPCombined[key], profitEachYear[key], color = plotColors[key], label = key, alpha = 0.8)
plt.xlabel("Min sell price for the day")
plt.ylabel("Profit for the day")
plt.title("Daily profit over min sell price for the day")
plt.legend()
plt.savefig("Daily profit over min sell price for the day")
plt.show()

listDiff: list[np.array] = []
for key in profitEachYear.keys():
    listDiff.append(maxSPCombined[key]-minSPCombined[key])
    plt.scatter(maxSPCombined[key]-minSPCombined[key], profitEachYear[key], color = plotColors[key], label = key, alpha = 0.8)
plt.xlabel("Diff sell price for the day")
plt.ylabel("Profit for the day")
plt.legend()
plt.title("Daily profit over diff in sell price for the day")
plt.savefig("Daily profit over diff in sell price for the day")
plt.show()

#plt.scatter(dailyMaxSP-dailyMinSP, dailyProfit)
for key in profitEachYear.keys():
    dailyDiffYearly = maxSPCombined[key]-minSPCombined[key]
    #betas: tuple = regression(dailyDiffYearly, profitEachYear[key])
    #yValues = betas[0] + betas[1] * xValues
    #profitMeanYear = (betas[0]+betas[1] * np.mean(dailyDiffYearly))
    plt.scatter(dailyDiffYearly, profitEachYear[key], color = plotColors[key], label = key, alpha = 0.8)
    #plt.plot(xValues, yValues, "--", color = plotColors[key], label = f"{betas[0]:.2f}+{betas[1]:.2f}x", alpha = 0.8)
    #print(f"The expected profit for the avrage day in year {key} is {profitMeanYear:.2f}")
    print(f"The real avarage daily profit for that year was {np.mean(profitEachYear[key]):.2f}")

plt.xlabel("Diff sell price for the day")
plt.ylabel("Profit for the day")
plt.plot(xValues, yValues, "--", color = "#8E9616")
#handles, labels = plt.gca().get_legend_handles_labels()
#newOrder = [0,2,4,6,8,1,3,5,7,9]
#plt.legend(
    #[handles[i] for i in newOrder],
    #[labels[i] for i in newOrder],
    #ncols = 2)
plt.legend()
plt.title("Linear regression between daily profit and diff in sell price")
plt.savefig("Linear regression between daily profit and diff in sell price")
plt.show()

plt.close()
print(ElPrice_mean_year)
print(ElPrice_mean_year["l_s"].to_list())


expectedProfit: list = [answer[0] + answer[1] * np.mean(value) for value in listDiff]
print(expectedProfit)
plt.bar(list(profitEachYear.keys()), expectedProfit, color = colorlist)
plt.grid(True, alpha = 0.5)
plt.ylabel("Expected profit")
plt.title("Expected daily profit for each year")
plt.savefig("Expected daily profit for each year")
plt.show()


#fit = smf.ols(dailyDiff, profitEachYear).fit()
#print(fit.summary())




#%%
seePlot = True
dayNumber = 0
showMultiple = False
print("Type a number (int) and get a plot for the SOC of that day")
print("Print: Stop, to break the loop, use + to get the next day and minus for the previus day")
while seePlot:
    answer = input(">")
    plt.close()
    if answer == "A":
        showMultiple = not showMultiple
        print(f"You have switch to {showMultiple=}")
    elif answer == "Stop":
        print("Stop")
        break
    elif answer == "+":
        print("In +")
        dayNumber += 1
    elif answer == "-":
        print("In -")
        dayNumber += -1
    elif answer.isnumeric():
        print("number")
        dayNumber = int(answer)
        dayNumber -= 1
    else:
        print("The input was not valid, only positive int or (+, -, Stop, A) is valid")
        print("If you want to end, type 'Stop'")
        continue
    print(f"{dayNumber+1=}")
    if dayNumber < 0 or dayNumber > 1825:
        print("There are no days in this range")
        print(f"{dayNumber-1}")
        print(f"dayNumber is set to 1")
        dayNumber = 1
    if dayNumber <= 365 and showMultiple:
        years = 2021
        for i in range(len(profitEachYear)):
            ElPrice_dayN = ElPrice.iloc[24 * (dayNumber) + (365*i):24*(dayNumber+1) + (365*i),ElPrice.columns.get_loc('l_s')].to_numpy()
            profit, charging, discharging, SOC = DayliOptimizer(ElPrice_dayN, params)
            plt.plot(SOC, label = f"{years+i}", color = colorlist, alpha = 0.8)
            print(f"{years + i}:{profit=:.2f} DKK", end= ", ")
            print(f"Energi bought: {charging.sum():.2f} kWh", end =", ")
            print(f"Energi sold: {discharging.sum():.2f} kWh")
        plt.title(f"SOC for day {dayNumber+1} in each year")
        plt.ylim(0.5,11.5)
        plt.legend(loc = "upper center", bbox_to_anchor=(0.5, 1), ncol= 5, fancybox = True, shadow =  True)
        plt.xticks(list(range(0,24)))
    else:
        ElPrice_dayN = ElPrice.iloc[24 * (dayNumber):24*(dayNumber+1),ElPrice.columns.get_loc('l_s')]
        print(ElPrice_dayN)
        ElPrice_dayN = ElPrice.iloc[24 * (dayNumber):24*(dayNumber+1),ElPrice.columns.get_loc('l_s')].to_numpy()
        print(len(ElPrice_dayN))
        profit, charging, discharging, SOC = DayliOptimizer(ElPrice_dayN, params)
        print(f"{profit=:.2f} DKK")
        print(len(SOC))
        print(charging)
        print(f"Energi bought {charging.sum()=:.2f} kWh")
        print(f"Energi sold {discharging.sum()=:.2f} kWh")
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.step(list(range(24)),ElPrice_dayN, color = color)
        #ax1.bar(list(range(0,24)),ElPrice_dayN, width = -0.8, color = color, align = 'edge')
        ax1.set_ylabel("El-Price", color =color)
        ax1.set_xlabel("Hours")
        ax1.set_xticks(list(range(0,24)))
        ax1.set_xlim(0,23)
        ax1.tick_params(axis='y', labelcolor = color)
        ax1.set_title(f"Day: {dayNumber}")
        color = 'tab:blue'
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel("SOC", color =color)
        ax2.tick_params(axis='y', labelcolor = color)
        ax2.set_yticks(list(range(1,10)))
        ax2.plot(SOC, color = color)
        #ax2.plot([0.5,SOC[0]],[-1,0], color = color)
    plt.grid(True)
    plt.show(block = False)
    #plt.pause(3)
    #plt.close()





# %%

