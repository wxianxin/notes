# Steven Wang
# 2017

################################################################################
################################################################################
# seaborn
import seaborn as sns
sns.set()
# To switch to seaborn defaults, simply call the set() function.

# sns.set_style("whitegrid")
# sns.set_style("dark")
# sns.set_style("white")
sns.set_style("darkgrid")
# sns.set(style="ticks")
# sns.set(style="ticks")

# set colors
# current_palette = sns.color_palette()
# sns.palplot(current_palette)

# sns.set_palette("husl")


# Scaling plot elements
# The four preset contexts, in order of relative size, are :
#   paper, notebook, talk, poster
sns.pairplot(ndf)
################################################################################
################################################################################


# Anatomy of Matplotlib

# For Jupyter/IPython notebooks, it's often easiest to use the Jupyter nbagg or notebook backend. This allows plots to be displayed and interacted with in the browser in a Jupyter notebook. Otherwise, figures will pop up in a separate GUI window.
# You can achieve this in two ways:
# 1. %matplotlib backend_name
# 2. matplotlib.use("backend_name")

import matplotlib
matplotlib.use('nbagg')

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Anatomy of a "Plot"
# Figure:  The top-level container in this hierarchy. It is the overall window/page that everything is drawn on. You can have multiple independent figures and Figures can contain multiple Axes.

# Axes: Most plotting ocurs on an Axes. The axes is effectively the area that we plot data on and any ticks/labels/etc associated with it. Usually we'll set up an Axes with a call to subplot (which places Axes on a regular grid), so in most cases, Axes and Subplot are synonymous.
# Each Axes has an XAxis and a YAxis. It is worth mentioning here to explain where the term Axes comes from.
# An Axes is made up of Axis objects and many other things.
# An Axes object must belong to a Figure (and only one Figure).

# Typically, you'll set up a Figure, and then add an Axes to it.
################################################################################

################################################################################
# create a figure
# fig = plt.figure()
# plt.show()

#  control the size of the figure through the figsize argument, which expects a tuple of (width, height) in inches.
# Twice as tall as it is wide:
fig = plt.figure(figsize=plt.figaspect(2.0))
plt.show()

################################################################################

################################################################################
# Axes
fig = plt.figure()
ax = fig.add_subplot(111) # Basically, 1 row and 1 column.
ax.set(xlim=[0.5, 4.5], ylim=[-2, 8], title='An Example Axes',
               ylabel='Y-Axis', xlabel='X-Axis')
plt.show()

# Note that the set method doesn't just apply to Axes; it applies to more-or-less all matplotlib objects.
# ax.set_*
# ax.set_xlim([0.5, 4.5])
# ax.set_ylim([-2, 8])
# ax.set_title('An Example Axes')
# ax.set_ylabel('Y-Axis')
# ax.set_xlabel('X-Axis')
################################################################################

################################################################################
# plot & scatter
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
ax.set_xlim(0.5, 4.5)
plt.show()
################################################################################

################################################################################
# Multiple Axes
fig, axes = plt.subplots(nrows=2, ncols=2)
# The axes object that was returned is a 2D numpy object array.
axes[0,0].set(title='Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Lower Left')
axes[1,1].set(title='Lower Right')

# To iterate over all items in a multidimensional numpy array, use the `flat` attribute
for ax in axes.flat:
        # Remove all xticks and yticks...
            ax.set(xticks=[])

            plt.show()


# # plt.subplots(): When it's called with no arguments, it creates a new figure with a single subplot.
# # Any time you see something like
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
# # You can replace it with:
#     fig, ax = plt.subplots()
################################################################################

################################################################################
# Jupyter/IPython magic functions
# %load path/file_name.py
# # the content of filename.py will be loaded in the next cell.
# %%writefile filename.py
# # To save the cell content back into a file add the cell-magic %%writefile filename.py at the beginning of the cell and run it. Beware that if a file with the same name already exists it will be silently overwritten.
# %magic
# %lsmagic
# # For general help on magic functions type "%magic" For a list of the available magic functions, use %lsmagic. For a description of any of them, type %magic_name?, e.g. '%cd?'.
################################################################################


EOF
# TBC
