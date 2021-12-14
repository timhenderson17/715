import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

pd.set_option('display.max_rows', 100)

# Pickling

pickling = 1
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

# MODIFY DATA DELETE ME BEFORE NIRUPAM SEES
#######
#######


plot = 1
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

# Heat map

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



