import pandas as pd
import matplotlib.pyplot as plt

#######
# UNCOMMENT IF NOT USING PICKLES

# df_bl = pd.read_csv('./back left facing teacher (small).csv')
# df_br = pd.read_csv('./back right facing teacher (caution tape).csv')
# df_fl = pd.read_csv('./front left facing teacher (close to entrance).csv')
# df_fr = pd.read_csv('./front right facing teacher (by exit).csv')

# pd.to_datetime(df_bl['Time'])
# pd.to_datetime(df_br['Time'])
# pd.to_datetime(df_fl['Time'])
# pd.to_datetime(df_fr['Time'])

# # so all data starts at exact same time
# # 7:48:33 PM
# df_fl = df_fl.iloc[61: , :]
# df_fr = df_fr.iloc[39: , :]
# df_bl = df_bl.iloc[14: , :]

# # so all data ends at exact same time
# # 12:15:53 PM
# df_fl = df_fl.iloc[:110550 , :]
# df_fr = df_fr.iloc[:110550 , :]
# df_bl = df_bl.iloc[:110550 , :]
# df_br = df_bl.iloc[:110550 , :]

# df_bl.to_pickle("./df_bl.pkl")
# df_br.to_pickle("./df_br.pkl")
# df_fl.to_pickle("./df_fl.pkl")
# df_fr.to_pickle("./df_fr.pkl")

#######

# Read pickles
df_bl = pd.read_pickle("./df_bl.pkl")
df_br = pd.read_pickle("./df_br.pkl")
df_fl = pd.read_pickle("./df_fl.pkl")
df_fr = pd.read_pickle("./df_fr.pkl")

fig, axes = plt.subplots(nrows=2, ncols=2)

ax_bl = df_bl.plot(ax=axes[0,0])
ax_br = df_br.plot(ax=axes[0,1])
ax_fl = df_fl.plot(ax=axes[1,0])
ax_fr = df_fr.plot(ax=axes[1,1])

ax_bl.set_ylim(0,1)
ax_br.set_ylim(0,1)
ax_fl.set_ylim(0,1)
ax_fr.set_ylim(0,1)

plt.show()
# plt.savefig('foo.png')