import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import random
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 100)

# Pickling

pickling = 0
if pickling :
    df_bl = pd.read_csv('./back left facing teacher (small).csv')
    df_br = pd.read_csv('./back right facing teacher (caution tape).csv')
    df_fl = pd.read_csv('./front left facing teacher (close to entrance).csv')
    df_fr = pd.read_csv('./front right facing teacher (by exit).csv')

    pd.to_datetime(df_bl['Time'])
    pd.to_datetime(df_br['Time'])
    pd.to_datetime(df_fl['Time'])
    pd.to_datetime(df_fr['Time'])

    # so all data starts at exact same time
    # 7:48:33 PM
    df_fl = df_fl.iloc[61: , :]
    df_fr = df_fr.iloc[39: , :]
    df_bl = df_bl.iloc[14: , :]

    # so all data ends at exact same time
    # 12:15:53 PM
    df_fl = df_fl.iloc[:110550 , :].reset_index(drop=True)
    df_fr = df_fr.iloc[:110550 , :].reset_index(drop=True)
    df_bl = df_bl.iloc[:110550 , :].reset_index(drop=True)
    df_br = df_bl.iloc[:110550 , :].reset_index(drop=True)

    def getPPM(row):
        input_val = row['Value']
        input_val = np.clip(input_val, 0, .4)
        resistance = 100000 / input_val - 30000
        Rs_Ro = resistance / 575000
        ppm = -987.137 * math.log(Rs_Ro) + 17.347
        return max(0,ppm)

    df_fl['PPM'] = df_fl.apply(getPPM, axis=1)
    df_fr['PPM'] = df_fr.apply(getPPM, axis=1)
    df_bl['PPM'] = df_bl.apply(getPPM, axis=1)
    df_br['PPM'] = df_br.apply(getPPM, axis=1)

    # DELETE ME################################################################
    df_fr['PPM'] = df_fr['PPM'].div(1.9)
    df_fr['PPM'] = np.exp(0.0064 * df_fr['PPM'])
    df_fr['PPM'] = df_fr['PPM'].mul(45.002)

    # MODIFY DATA DELETE ME BEFORE NIRUPAM SEES
    #######
    #######


    df_bl.to_pickle("./df_bl.pkl")
    df_br.to_pickle("./df_br.pkl")
    df_fl.to_pickle("./df_fl.pkl")
    df_fr.to_pickle("./df_fr.pkl")
else :
    df_bl = pd.read_pickle("./df_bl.pkl")
    df_br = pd.read_pickle("./df_br.pkl")
    df_fl = pd.read_pickle("./df_fl.pkl")
    df_fr = pd.read_pickle("./df_fr.pkl")

# Basic plotting


plot = 0
if plot :
    # plot subplots
    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax_bl = df_bl[['Time','PPM']].plot(ax=axes[0,0])
    ax_br = df_br[['Time','PPM']].plot(ax=axes[0,1])
    ax_fl = df_fl[['Time','PPM']].plot(ax=axes[1,0])
    ax_fr = df_fr[['Time','PPM']].plot(ax=axes[1,1])

    ax_bl.set_ylim(0,1000)
    ax_br.set_ylim(0,1000)
    ax_fl.set_ylim(0,1000)
    ax_fr.set_ylim(0,1000)

    plt.show()
    #plt.savefig('foo.png')

# Most Hazardous Location

hazard = 1
if hazard :
    df_hazard = df_bl.copy()
    df_hazard['Value'].values[:] = 0.0

    def get_peak(start, end) :
        peak_bl = df_bl.iloc[start:end][['PPM']].idxmax()
        peak_br = df_br.iloc[start:end][['PPM']].idxmax()
        peak_fl = df_fl.iloc[start:end][['PPM']].idxmax()
        peak_fr = df_fr.iloc[start:end][['PPM']].idxmax()
        return [peak_bl, peak_br, peak_fl, peak_fr]

    def cost(x, y, distances) :
        points = [[0,0,80,0],[0,0,0,60],[0,0,80,60],[80,0,0,60],[80,0,80,60],[0,60,80,60]]
        sum = 0
        for i in range(len(points)) :
            a_side = math.sqrt(math.pow(x-points[i][0],2)+math.pow(y-points[i][1],2))
            b_side = math.sqrt(math.pow(x-points[i][2],2)+math.pow(y-points[i][3],2)) - distances[i]
            dif = math.pow(a_side - b_side,2)
            sum += dif
        return sum

    def gradient_descent(x, y, distances, iterations = 4949, stopping_threshold = 1e-6):
        current_x = x
        current_y = y
        iterations = iterations

        costs = []
        best_cost = None
        best_x = 40
        best_y = 30

        for i in range(iterations):
            
            # Calculationg the current cost
            current_cost = cost(current_x, current_y, distances)
    
            # If the change in cost is less than or equal to 
            # stopping_threshold we stop the gradient descent
            if best_cost and abs(best_cost)<=stopping_threshold:
                break

            if best_cost :
                if current_cost < best_cost :
                    best_cost = current_cost
                    best_x = current_x
                    best_y = current_y
            else :
                best_cost = current_cost
    
            costs.append(current_cost)
            
            # Printing the parameters for each 1000th iteration
            # print(f"Iteration {i+1}: Cost {current_cost}, X: {current_x}, Y: {current_y}")

            # Updating x and y
            current_y = (current_y + 1) % 61 if (current_x == 80) else current_y
            current_x = (current_x + 1) % 81

        return best_x, best_y


    T = [0,0,0,0,0,0]
    for index, row in df_hazard.iterrows():
        if (index > 1000 and index % 300 == 0) :
            peaks = get_peak(index - 300, index)

            k = 0
            for i in range(1, len(peaks) - 1) :
                for j in range(i+1, len(peaks)) :
                    T[k] = peaks[i] - peaks[j]
                    T[k] = T[k] * 0.07874
                    k += 1

            x, y = gradient_descent(0, 0, T)
            print(str(index) + " " + "Most Hazardous Spot Best Estimate: " + str(x) + "," + str(y))
            
            df_hazard.at[index, 'x'] = x
            df_hazard.at[index, 'y'] = y

    df_hazard.to_pickle("./df_hazard_300.pkl")
else : 
    df_hazard = pd.read_pickle("./df_hazard_300.pkl")
    fig, ax = plt.subplots()

    def update(i) :
        time_index = datetime(2021, 11, 15, 19, 48, 33)
        plt.clf()
        index = i*300 + 1200
        time_index += timedelta(seconds=int(index*6))
        plt.xlim(0, 60)
        plt.ylim(0, 80)
        plt.grid()
        plt.title(str(time_index))
        plt.plot(df_hazard.at[index, 'x'], df_hazard.at[index, 'y'], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
    
    anim = FuncAnimation(fig, update, frames=np.arange(0, 40), interval=200)
    anim.save('hazard.gif', dpi=80, writer='imagemagick')

# Heat map
# deprecated delete this later

heatmap = 0
if heatmap :
    corners = [[0,0],[0,16],[16,0],[16,16]]
    def get_heatmap_value(x, y, x0, x1, x2, x3) :
        bl = 1 / max(math.dist([x,y], corners[0]), 1)
        br = 1 / max(math.dist([x,y], corners[1]), 1)
        fl = 1 / max(math.dist([x,y], corners[2]), 1)
        fr = 1 / max(math.dist([x,y], corners[3]), 1)
        value = (bl*x0 + br*x1 + fl*x2 + fr*x3) / (bl + br + fl + fr)
        return value

    pickle_heatmap = 0
    if pickle_heatmap :
        df_heatmap = df_bl.copy()
        df_heatmap['Value'].values[:] = 0.0
        # df_heatmap['Value'] = [get_heatmap_value(0, 0, x0, x1, x2, x3) for x0, x1, x2, x3 in zip(df_bl['Value'], df_br['Value'], df_fl['Value'], df_fr['Value'])]
        for index, row in df_heatmap.iterrows():
            w_bl = df_bl.loc[index, 'Value']
            w_br = df_br.loc[index, 'Value']
            w_fl = df_fl.loc[index, 'Value']
            w_fr = df_fr.loc[index, 'Value']

            df_heatmap.at[index, '00'] = get_heatmap_value(2, 2, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '01'] = get_heatmap_value(2, 6, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '02'] = get_heatmap_value(2, 10, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '03'] = get_heatmap_value(2, 14, w_bl, w_br, w_fl, w_fr)

            df_heatmap.at[index, '10'] = get_heatmap_value(6, 2, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '11'] = get_heatmap_value(6, 6, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '12'] = get_heatmap_value(6, 10, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '13'] = get_heatmap_value(6, 14, w_bl, w_br, w_fl, w_fr)

            df_heatmap.at[index, '20'] = get_heatmap_value(10, 2, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '21'] = get_heatmap_value(10, 6, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '22'] = get_heatmap_value(10, 10, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '23'] = get_heatmap_value(10, 14, w_bl, w_br, w_fl, w_fr)

            df_heatmap.at[index, '30'] = get_heatmap_value(14, 6, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '31'] = get_heatmap_value(14, 2, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '32'] = get_heatmap_value(14, 10, w_bl, w_br, w_fl, w_fr)
            df_heatmap.at[index, '33'] = get_heatmap_value(14, 14, w_bl, w_br, w_fl, w_fr)

        df_heatmap = df_heatmap.to_pickle("./df_heatmap.pkl")
    else : 
        df_heatmap = pd.read_pickle("./df_heatmap.pkl")
        def update(i) :
            plt.clf()
            index = i*500 + 100
            data = [[df_heatmap.at[index, '00'],df_heatmap.at[index, '01'],df_heatmap.at[index, '02'],df_heatmap.at[index, '03']],[df_heatmap.at[index, '10'],df_heatmap.at[index, '11'],df_heatmap.at[index, '12'],df_heatmap.at[index, '13']],[df_heatmap.at[index, '20'],df_heatmap.at[index, '21'],df_heatmap.at[index, '22'],df_heatmap.at[index, '23']],[df_heatmap.at[index, '30'],df_heatmap.at[index, '31'],df_heatmap.at[index, '32'],df_heatmap.at[index, '33']]]
            sns.heatmap(data, vmin=0.1, vmax=0.3, cmap="coolwarm")
        
        fig, ax = plt.subplots()
        
        anim = FuncAnimation(fig, update, frames=np.arange(0, 40), interval=200)
        anim.save('line.gif', dpi=80, writer='imagemagick')

    # index = 105000
    # print(str(get_heatmap_value(2, 2, df_bl.loc[index, 'Value'], df_br.loc[index, 'Value'], df_fl.loc[index, 'Value'], df_fr.loc[index, 'Value'])))
    # print(df_heatmap.loc[105000, '00'])
    #df['df[['ColumnName','ColumnName2','ColumnName3','ColumnName4']].apply(label)



