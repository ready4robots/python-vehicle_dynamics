#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[3]:


SCENARIO_NAME = "Adult Pedestrian Crossing"


run1 = pd.read_csv('Olli-slow-pedestrian_2020_08_11_134658.csv')
run2 = pd.read_csv('Olli-slow-pedestrian_2020_08_11_135224.csv')
run3 = pd.read_csv('Olli-slow-pedestrian_2020_08_11_141723.csv')
run4 = pd.read_csv('Olli-slow-pedestrian_2020_08_11_142751.csv')
run5 = pd.read_csv('Olli-slow-pedestrian_2020_08_11_143000.csv')

run1.columns = run1.columns.str.replace("/",'')
run1.columns = run1.columns.str.replace(" ",'')
run1.columns = run1.columns.str.replace("_",'')
run1.columns = run1.columns.str.replace("[",'')
run1.columns = run1.columns.str.replace("]",'')
run1.columns = run1.columns.str.replace(".1",'')
run1.columns = run1.columns.str.replace("-",'')

run2.columns = run2.columns.str.replace("/",'')
run2.columns = run2.columns.str.replace(" ",'')
run2.columns = run2.columns.str.replace("_",'')
run2.columns = run2.columns.str.replace("[",'')
run2.columns = run2.columns.str.replace("]",'')
run2.columns = run2.columns.str.replace(".1",'')
run2.columns = run2.columns.str.replace("-",'')

run3.columns = run3.columns.str.replace("/",'')
run3.columns = run3.columns.str.replace(" ",'')
run3.columns = run3.columns.str.replace("_",'')
run3.columns = run3.columns.str.replace("[",'')
run3.columns = run3.columns.str.replace("]",'')
run3.columns = run3.columns.str.replace(".1",'')
run3.columns = run3.columns.str.replace("-",'')

run4.columns = run4.columns.str.replace("/",'')
run4.columns = run4.columns.str.replace(" ",'')
run4.columns = run4.columns.str.replace("_",'')
run4.columns = run4.columns.str.replace("[",'')
run4.columns = run4.columns.str.replace("]",'')
run4.columns = run4.columns.str.replace(".1",'')
run4.columns = run4.columns.str.replace("-",'')

run5.columns = run5.columns.str.replace("/",'')
run5.columns = run5.columns.str.replace(" ",'')
run5.columns = run5.columns.str.replace("_",'')
run5.columns = run5.columns.str.replace("[",'')
run5.columns = run5.columns.str.replace("]",'')
run5.columns = run5.columns.str.replace(".1",'')
run5.columns = run5.columns.str.replace("-",'')


# In[4]:


table = pd.concat([run1, run2, run3, run4, run5],  keys=['run1', 'run2', 'run3','run4','run5',], names=['RunCount', 'Row ID'], sort=False)


# In[5]:


# all runs - names
run_names = ['run1', 'run2', 'run3', 'run4', 'run5']

# list with individual run DFs
all_runs = [run1, run2, run3, run4, run5]


# In[6]:


fig, ax = plt.subplots(figsize=(15,7))

table.groupby(['Times','RunCount']).sum()['RangPosRes;MAXm'].unstack().plot(ax=ax)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Range Resultant - meters')
plt.title('Olli Adult Crossing: Relative Distance (Range)')
fig.savefig('Adult-Crossing-Range.png')

fig, ax = plt.subplots(figsize=(15,7))
plt.ylim(0, 1)
table.groupby(['Times','RunCount']).sum()['RangPosRes;MAXm'].unstack().plot(ax=ax)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Range Resultant - meters')
plt.title('Adult Crossing: Relative Distance (Range) - Zoom')
fig.savefig('Adult-Crossing-Range-Zoom.png')

fig, ax = plt.subplots(figsize=(15,7))
table.groupby(['Times','RunCount']).sum()['TargeSpeed2D;MAXms'].unstack().plot(ax=ax)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Target Speed - m/s')
plt.title('Adult Crossing: Ego Speed')
fig.savefig('Adult-Crossing-Target-Speed.png')

fig, ax = plt.subplots(figsize=(15,7))
table.groupby(['Times','RunCount']).sum()['HunterSpeed2D;MAXms'].unstack().plot(ax=ax)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Target Speed - m/s')
plt.title('Adult Crossing: Actor Speed')
fig.savefig('Adult-Crossing-Ped-Speed.png')


# # 3-in-1 Plots for Individual Runs

# In[10]:


# Plot 3-in-1 figure with charts for individual runs - save as png
def plot_one_run_charts(cdf, run_name='run'):
    '''
        Plot Speeds, Accels, Ego Pitch for given run - on a single figure
        cdf: df for single run
        run_name: name of the run
    '''
    
    cdf = cdf.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle',
                    'RangTimeToCollisionForwardWithAccel;MAXs' : 'TTC'
                   })
    
    ## Variables
    x_time    = cdf['Time']
    # Ego vars
    ego_range = cdf['Range']
    ego_speed = cdf['Hunter Velocity']
    ego_accel = cdf['Ego Forward Accel']
    ego_pitch = cdf['Ego Pitch Angle']
    ego_ttc   = cdf['TTC']
    
    # target vars
    t_speed = cdf['Target Velocity']
    t_accel = cdf['Target Forward Accel']
    
    ## Processing - find Min Dist and time for Min Dist
    
    # Min Distance - use with caution
    D_MIN = ego_range.min()
    LABEL_MIN_DIST = "Min Dist: " + str(D_MIN)
    
    # time at D_MIN - use with caution - could be outside 
    X_TIME_MIN = x_time[ego_range.idxmin()]
    
    # TTC cleanup - remove outliers (overflows, underflows)
    ttc_trim = np.where(ego_ttc > 25, np.nan, ego_ttc)       # remove TTC above 25 secs
    ttc_trim = np.where(ego_ttc < -5, np.nan, ttc_trim)      # remove < -5 seconds
    ttc_series = pd.Series(ttc_trim)
    
    # TTC: rolling mean for 5 samples...(ignores NaN)
    ttc_rolling5 = ttc_series.rolling(5, min_periods=2).mean()
    # ...interpolate to smooth out
    ttc_rolling5 = ttc_rolling5.interpolate('polynomial', order=2)
    
    ### Plots: 3-in-1 figure
    fig, axs = plt.subplots(3, 1, figsize=(12,14), constrained_layout=True)
    
    # 1 - Speeds
    axs[0].plot(x_time, ego_speed, x_time, t_speed)
    axs[0].set_xlabel('Time - seconds')
    axs[0].set_ylabel('Speed - m/s')
    
    # min dist line
    axs[0].axvline(X_TIME_MIN, color='red', linestyle='--') # vertical line
    #  Dmin label
    axs[0].text(X_TIME_MIN + 0.5, 1.5, f'Min Dist: {D_MIN}m\n\nat t = {X_TIME_MIN}s' )
    
    r_ax = axs[0].twinx()  # right axis for Range
    r_ax.plot(x_time, ego_range, color='green')
    r_ax.set_ylabel('Range - meters')
    r_ax.tick_params(axis='y', labelcolor='green')
    
    lines = axs[0].get_lines() + r_ax.get_lines()
    axs[0].legend(lines, ['Ego', 'Target', 'Min Dist', 'Range (RHS)'])
    
    title = f"{SCENARIO_NAME} - {run_name}: Speeds"
    axs[0].set_title(title)

    
    # 2 - Accelerations
    axs[1].plot(x_time, ego_accel, x_time, t_accel)
    axs[1].set_xlabel('Time - seconds')
    axs[1].set_ylabel('Accel - m/s^2')
    hj-0
    r_ax = axs[1].twinx()  # right axis for Range
    r_ax.plot(x_time, ego_range, color='green')
    r_ax.set_ylabel('Range - meters')
    r_ax.tick_params(axis='y', labelcolor='green')

    lines = axs[1].get_lines() + r_ax.get_lines()
    axs[1].legend(lines, ['Ego', 'Target', 'Range (RHS)'])
    title = f"{SCENARIO_NAME} - {run_name}: Linear Accelerations"
    axs[1].set_title(title)
    # min dist line
    axs[1].axvline(X_TIME_MIN, color='red', linestyle='--') # vertical line

    
    # 3 - Pitch Angle
    axs[2].plot(x_time, ego_pitch)
    axs[2].set_xlabel('Time - seconds')
    axs[2].set_ylabel('Pitch Angle - degrees')
    
    r_ax = axs[2].twinx()  # right axis for Range
    r_ax.plot(x_time, ego_range, color='green')
    r_ax.set_ylabel('Range - meters')
    r_ax.tick_params(axis='y', labelcolor='green')

    
    lines = axs[2].get_lines() + r_ax.get_lines()
    axs[2].legend(lines, ['Ego', 'Range (RHS)'])
    title = f"{SCENARIO_NAME} - {run_name}: Ego Pitch"
    axs[2].set_title(title)
    # min dist line
    axs[2].axvline(X_TIME_MIN, color='red', linestyle='--') # vertical line

    
    # Save figure
    figname = f"pic_{SCENARIO_NAME}_{run_name}.png"
    fig.savefig(figname)
    
    return


# In[11]:


## Plot Min Distance for all runs on 1 Chart
def plot_min_distance_chart(runs, run_names):
    ''' Plot Minimum Distance for given list of runs
        runs =  list of dataframes for each run
        run_names = list of names for runs (e.g. 'Run 1', 'Run 3', ...)
    '''
    
    list_min_dist = []
    
    # get min dist for each run
    for ix, run_df in enumerate(runs):
        # var Range
        r_range = run_df['RangPosRes;MAXm']
        r_min_dist = r_range.min() #min dist
        list_min_dist.append(r_min_dist) # store
    
    # new small data frame
    rdf = pd.DataFrame({'Min Distance': list_min_dist}, index=run_names)
    avg_min_dist = round(np.mean(list_min_dist), 3)
    
    # plot bar chart
    #fig, _ = plt.subplots(figsize=(6,4))
    ax = rdf.plot.bar()
    ax.set_title(f"{SCENARIO_NAME} - Minimum Distances for Runs")
    ax.set_ylabel('Range (meters)')
    # mean line
    ax.axhline(avg_min_dist, color='red', linestyle='--')
    ax.text(x=1, y=1, s='Mean Min Dist = ' + str(avg_min_dist) + 'm')
    
    # Save fig
    figname = f'pic_{SCENARIO_NAME}_min_dists.png'
    plt.savefig(figname)
    


# ## Plot the single charts for each run

# In[13]:


## plot 3-in-1 figures for all runs

for ix, run_df in enumerate([run1, run2, run3, run4, run5]):
    
    # plot 3-in-1 figure for each run
    plot_one_run_charts(run_df, run_names[ix])   # run name is passed


# In[ ]:





# ## plot min distance for all runs together

# In[14]:


# DF for all runs individually
all_runs = [run1, run2, run3, run4, run5]

# plot Min dist
plot_min_distance_chart(all_runs, run_names)


# In[ ]:





# In[15]:


#run1 plots: Velocity, Accel, Pitch, Range
run1b = run1.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle'            
                   })

fig, ax = plt.subplots(figsize=(15,7))
ax=run1b.plot.line(x='Time', y= ['Target Velocity', 'Hunter Velocity','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Speed - m/s')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines])
plt.title('Adult Crossing (run1): Velocities')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run1b.plot.line(x='Time', y= ['Target Forward Accel', 'Ego Forward Accel','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Linear Acceleration - m/s^2')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run1): Linear Accelerations')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run1b.plot.line(x='Time', y= ['Ego Pitch Angle','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Pitch Angle - degrees')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run1): Ego Vehicle Pitch Angle')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')


# In[16]:


#run2 plots: Velocity, Accel, Pitch, Range
run2b = run2.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle'            
                   })

fig, ax = plt.subplots(figsize=(15,7))
ax=run2b.plot.line(x='Time', y= ['Target Velocity', 'Hunter Velocity','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Speed - m/s')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines])
plt.title('Adult Crossing (run2): Velocities')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run2b.plot.line(x='Time', y= ['Target Forward Accel', 'Ego Forward Accel','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Linear Acceleration - m/s^2')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run2): Linear Accelerations')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run2b.plot.line(x='Time', y= ['Ego Pitch Angle','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Pitch Angle - degrees')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run2): Ego Vehicle Pitch Angle')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')


# In[13]:


#run3 plots: Velocity, Accel, Pitch, Range
run3b = run3.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle'            
                   })

fig, ax = plt.subplots(figsize=(15,7))
ax=run3b.plot.line(x='Time', y= ['Target Velocity', 'Hunter Velocity','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Speed - m/s')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines])
plt.title('Adult Crossing (run3): Velocities')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run3b.plot.line(x='Time', y= ['Target Forward Accel', 'Ego Forward Accel','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Linear Acceleration - m/s^2')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run3): Linear Accelerations')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run3b.plot.line(x='Time', y= ['Ego Pitch Angle','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Pitch Angle - degrees')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run3): Ego Vehicle Pitch Angle')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')


# In[14]:


#run4 plots: Velocity, Accel, Pitch, Range
run4b = run4.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle'            
                   })

fig, ax = plt.subplots(figsize=(15,7))
ax=run4b.plot.line(x='Time', y= ['Target Velocity', 'Hunter Velocity','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Speed - m/s')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines])
plt.title('Adult Crossing (run4): Velocities')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run4b.plot.line(x='Time', y= ['Target Forward Accel', 'Ego Forward Accel','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Linear Acceleration - m/s^2')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run4): Linear Accelerations')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run4b.plot.line(x='Time', y= ['Ego Pitch Angle','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Pitch Angle - degrees')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run4): Ego Vehicle Pitch Angle')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')


# In[15]:


#run5 plots: Velocity, Accel, Pitch, Range
run5b = run5.rename(columns=
                   {'Times': 'Time', 
                    'HunterSpeed2D;MAXms': 'Target Velocity',
                    'TargeSpeed2D;MAXms': 'Hunter Velocity',
                    'HunterAccelForward;MAXms' : 'Target Forward Accel',
                    'TargeAccelForward;MAXms': 'Ego Forward Accel',
                    'RangPosRes;MAXm':'Range', 
                    'TargeAnglePitch;MAX' : 'Ego Pitch Angle'            
                   })

fig, ax = plt.subplots(figsize=(15,7))
ax=run5b.plot.line(x='Time', y= ['Target Velocity', 'Hunter Velocity','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Speed - m/s')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines])
plt.title('Adult Crossing (run5): Velocities')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run5b.plot.line(x='Time', y= ['Target Forward Accel', 'Ego Forward Accel','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Linear Acceleration - m/s^2')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run5): Linear Accelerations')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

fig, ax = plt.subplots(figsize=(15,7))
run5b.plot.line(x='Time', y= ['Ego Pitch Angle','Range'],secondary_y='Range',ax=ax,mark_right=False)
ax.set_xlabel('Time - seconds')
ax.set_ylabel('Pitch Angle - degrees')
ax.right_ax.set_ylabel('Range - meters')
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines],loc='lower left')
plt.title('Adult Crossing (run5): Ego Vehicle Pitch Angle')
#fig.savefig('Adult-Crossing-Ped-Speed.pdf')

