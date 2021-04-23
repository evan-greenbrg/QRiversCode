import os
from itertools import islice
import numpy
import pandas
from scipy import spatial
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from PyRivers.Migration import channelMigration


RIVER = 'red'
I = 1


class PointPicker(object):
    def __init__(self, ax, df):
        self.ax = ax
        self.df = df
        self.tree = spatial.KDTree(df[['x', 'ydetrend']])
        self.events = []
        self.points = []
        self.idxs = []

    def clear(self, event):
        self.events = []
        # Remove all plotted picked points
        for p in self.points:
            p.remove()
        self.points = []
        print('Cleared')

    def undo(self, event):
        self.events.pop(-1)
        undo_p = self.points.pop(-1)
        undo_p.remove()
        event.canvas.draw()
        print('Undid')

    def done(self, event):
        plt.close('all')
        print('All Done')

    def findIndex(self, event):
        distance, neighbor = self.tree.query(
            [self.x, self.y],
            1
        )
        self.idxs.append(neighbor)

    def __call__(self, event):
        self.event = event

        if not event.dblclick:
            return 0

        self.x, self.y = event.xdata, event.ydata
        self.events.append((self.x, self.y))
        self.events = list(set(self.events))

        if self.x is not None:
            self.findIndex(event)
            self.points.append(self.ax.scatter(self.x, self.y))
            event.canvas.draw()

def window(seq, n=3):
    """
    Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


# Powder
if RIVER == 'red':
    pairs = [
         (1986, 2020),
         (1988, 2020),
         (1990, 2020),
         (1992, 2020),
         (1994, 2020),
         (1996, 2020),
         (1998, 2020),
         (2000, 2020),
         (2002, 2020),
         (2004, 2020),
         (2006, 2020),
         (2008, 2020),
         (2010, 2020),
         (2014, 2020),
         (2016, 2020),
         (2018, 2020),
    ]
elif RIVER == 'beni':
    # Beni
     pairs = [
         (1986, 1992),
         (1992, 1998),
         (1998, 2004),
         (2004, 2010),
         (2010, 2014),
         (2014, 2018),
     ]
elif RIVER == 'itui':
    pairs = [
        (1984, 1990),
        (1990, 1996),
        (2002, 2008),
        (2008, 2016),
        (2016, 2020),
    ]
elif RIVER== 'ica':
    pairs = [
        (2004, 2010),
        (2010, 2016),
        (2016, 2020),
    ]

else:
    # Red 
     pairs = [
         (1986, 2008),
         (1988, 2020),
         (1990, 2020),
         (1992, 2020),
         (1994, 2020),
         (1996, 2020),
         (1998, 2020),
         (2000, 2020),
         (2002, 2020),
         (2004, 2020),
         (2006, 2020),
         (2008, 2020),
         (2010, 2020),
         (2014, 2020),
         (2016, 2020),
         (2018, 2020),
     ]
river = RIVER
clroot = f'/Users/greenberg/Documents/PHD/Projects/BarT/LinuxFiles/riverData/{river}/data/'
for pair in pairs:
    print(pair)
    # Get migration stats
    t1 = pair[0] 
    t2 = pair[1] 
    dt = t2 - t1

    if river == 'powder':
        cutoffthresh = 1000
        crosslen=80
    elif river == 'beni':
        cutoffthresh = 3000 
        crosslen=400
    elif river == 'itui':
        cutoffthresh = 1000 
        crosslen=100
    else:
        cutoffthresh = 4000
        crosslen=600

    df = channelMigration(
        clroot, 
        t1, 
        t2, 
        river, 
        cutoffthresh=cutoffthresh,
        smoothing=3,
        crosslen=crosslen,
        xcolumn='easting',
        ycolumn='northing'
    )

    # Turn it into velocities
    df['Vx'] = df['Xmigration'] / dt
    df['Vy'] = df['Ymigration'] / dt
    df['Vm'] = df['MagMigration'] / dt

    # # Manual Pick bar apexes
    # fig = plt.figure()
    # t = plt.gca()
    # pl = plt.plot(df['x'], df['ydetrend'])
    # PP = PointPicker(t, df)
    # 
    # axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
    # bclear = Button(plt.gca(), 'Clear')
    # bclear.on_clicked(PP.clear)
    # 
    # axundo = plt.axes([0.1, 0.0, 0.1, 0.1])
    # bundo = Button(plt.gca(), 'Undo')
    # bundo.on_clicked(PP.undo)
    # 
    # axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
    # bdone = Button(plt.gca(), 'Done')
    # bdone.on_clicked(PP.done)
    # 
    # fig.canvas.mpl_connect('button_press_event', PP)
    # plt.show()

    DISTANCE = 10
    distance = DISTANCE
    peaks, _ = signal.find_peaks(df['ydetrend'], distance=distance)
    troughs, _ = signal.find_peaks(-df['ydetrend'], distance=distance)
    bar_idxs = numpy.sort(numpy.concatenate((peaks, troughs)))

#    break
#    plt.plot(df['x'], df['ydetrend'])
#    plt.scatter(df.iloc[bar_idxs]['x'], df.iloc[bar_idxs]['ydetrend'])
#    plt.show()

    data = {
        'i': [],
        'x': [],
        'y': [],
        'ydetrend': [],
        'Vxb': [],
        'Vyb': [],
        'Vmb': [],
        'Hbf': [],
        'Lbf': []
    }
    for i, bars in enumerate(window(bar_idxs)):
        bar = df.iloc[min(bars):max(bars)]
        bar_apex = df.iloc[bars[1]]

        # Get lbf and hbf
        pair0 = [df.iloc[bars[0]]['x'], df.iloc[bars[1]]['x']]
        pair1 = [df.iloc[bars[1]]['x'], df.iloc[bars[2]]['x']]
        data['Lbf'].append(
            numpy.mean([
                numpy.max(pair0) - numpy.min(pair1),
                numpy.max(pair1) - numpy.min(pair1),
            ])
        )
        pair2 = [df.iloc[bars[0]]['y'], df.iloc[bars[1]]['y']]
        pair3 = [df.iloc[bars[2]]['y'], df.iloc[bars[1]]['y']]
        data['Hbf'].append(
            numpy.mean([
                numpy.max(pair2) - numpy.min(pair2),
                numpy.max(pair3) - numpy.min(pair3),
            ])
        )

        # Calculate bar statistics
        data['i'].append(i)
        data['x'].append(bar_apex['x'])
        data['y'].append(bar_apex['y'])
        data['ydetrend'].append(bar_apex['ydetrend'])
        data['Vxb'].append(bar['Vx'].mean())
        data['Vyb'].append(bar['Vy'].mean())
        data['Vmb'].append(bar['Vm'].mean())

    bar_df = pandas.DataFrame(data)
    bar_df['Db'] = bar_df['Vxb'] + bar_df['Vyb']
    bar_df['theta'] = bar_df['Db'] / bar_df['Vxb']

    Hbf = bar_df['Hbf'].median()
    Lbf = bar_df['Lbf'].median()

    Tm = Lbf / bar_df['Vxb']
    bar_df['Tm'] = bar_df['Lbf'] / bar_df['Vxb']
    Tma = Tm.median()

    Td = Hbf / bar_df['Db']
    bar_df['Td'] = bar_df['Hbf'] / bar_df['Db']
    Tda = Td.median()

    river = RIVER 
    clname = f'{river}_{t1}_migration_df.csv'
    barname = f'{river}_{t1}_bar_df.csv'
#    clname = f'idx{I}/{river}_{t1}_migration_df.csv'
#    barname = f'idx{I}/{river}_{t1}_bar_df.csv'


    oroot = f'/Users/greenberg/Documents/PHD/Projects/BarT/Analyses/timeSensitivity'
    aoutpath = os.path.join(oroot, str(t1), clname)
    bar_aoutpath = os.path.join(oroot, str(t1), barname)
    df.to_csv(aoutpath)
    bar_df.to_csv(bar_aoutpath)

    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(df['x'], df['ydetrend'])
    # axs[1].plot(df['x'], df['Vx'])
    # axs[2].plot(df['x'], df['Vy'])
    # plt.show()

