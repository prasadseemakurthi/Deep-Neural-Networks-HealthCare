from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from textwrap import wrap

# define colors for positive risk (red) and negative risk (blue)
REDFACE = '#DE2D26'
BLUEFACE = '#3182BD'
REDEDGE = '#DE2D26'
BLUEEDGE = '#3182BD'
MEDIAN = '#000000'
WHISKER = '#AAAAAA'
POINTS = '#000000'
GRID = '#BBBBBB'

# layout constants for boxplot
BOX_HSPACE = 0.15
BOX_VSPACE = 0.4
BOX_FH = 5  # boxplot figure width
BOX_FW = 8  # boxplot figure height
JITTER = 0.08
BOX_FONT = 8

# layout constants for pairwise feature plot
PAIR_FW = 10
PAIR_SPACING = 0.1

# layout constants for survival plot
SURV_FW = 10
SURV_FH = 6
SURV_HSPACE = 0.1
SURV_VSPACE = 0.1
SURV_FONT = 8


def RankedBar(Profile, Symbols, Types, XLabel=None, YLabel=None):
	"""
	Generates a bar plot of feature gradients or enrichment scores ranked by
	magnitude.

	Parameters:
	----------
	Profile : array_like
	Numpy array containing 1-dimensional feature/sample gradients or enrichment
	scoresobtained.

	Symbols : array_like
	List containing strings describing features in profile.

	Types : array_like
	List containing strings describing feature types (e.g. CNV, Mut, Clinical).

	XLabel : string
	Label for y axis. Default value = None

	YLabel : string
	Label for y axis. Default value = None

	Returns:
	--------
	Figure : figure handle
		Handle to figure used for saving image to disk i.e.
		Figure.savefig('heatmap.pdf')

	Notes:
	------
	Features are displayed in the order they are provided. Any sorting should
	happen prior to calling.
	"""

	# generate figure and add axes
	Figure = plt.figure(figsize=(BOX_FW, BOX_FH), facecolor='white')
	Axes = Figure.add_axes([BOX_HSPACE, BOX_VSPACE,
							1-BOX_HSPACE, 1-BOX_VSPACE],
						   frame_on=False)
	Axes.set_axis_bgcolor('white')

	# generate bars
	Bars = Axes.bar(np.linspace(1, len(Profile), len(Profile)), Profile,
					align='center')

	# modify box styling
	for i, bar in enumerate(Bars):
		if Profile[i] <= 0:
			bar.set(color=BLUEEDGE, linewidth=2)
			bar.set(facecolor=BLUEFACE)
		else:
			bar.set(color=REDEDGE, linewidth=2)
			bar.set(facecolor=REDFACE)

	# set limits
	Axes.set_ylim(1.05 * Profile.min(), 1.05 * Profile.max())

	# format x axis
	if XLabel is not None:
		plt.xlabel(XLabel)
	plt.xticks(np.linspace(1, len(Profile), len(Profile)),
			   [Symbols[i] + " _" + Types[i] for i in np.arange(len(Profile))],
			   rotation='vertical', fontsize=BOX_FONT)
	Axes.set_xticks(np.linspace(1.5, len(Profile)-0.5,
								len(Profile)-1), minor=True)
	Axes.xaxis.set_ticks_position('bottom')

	# format y axis
	if YLabel is not None:
		plt.ylabel(YLabel)
	Axes.yaxis.set_ticks_position('left')

	# add grid lines and zero line
	Axes.xaxis.grid(True, color=GRID, linestyle='-', which='minor')
	plt.plot([0, len(Profile)+0.5], [0, 0], color='black')

	return Figure


def RankedBox(Gradients, Symbols, Types, XLabel=None, YLabel=None):
	"""
	Generates boxplot series of feature gradients ranked by absolute magnitude.

	Parameters:
	----------
	Gradients : array_like
	Numpy array containing feature/sample gradients obtained by RiskCohort.
	Features are in columns and samples are in rows.

	Symbols : array_like
	List containing strings describing features.

	Types: array_like
	List containing strings describing feature types (e.g. CNV, Mut, Clinical).

	XLabel : string
	Label for y axis. Default value = None

	YLabel : string
	Label for y axis. Default value = None

	Returns:
	--------
	Figure : figure handle
		Handle to figure used for saving image to disk i.e.
		Figure.savefig('heatmap.pdf')

	Notes:
	------
	Features are displayed in the order they are provided. Any sorting should
	happen prior to calling.
	"""

	# generate figure and add axes
	Figure = plt.figure(figsize=(BOX_FW, BOX_FH), facecolor='white')
	Axes = Figure.add_axes([BOX_HSPACE, BOX_VSPACE,
							1-BOX_HSPACE, 1-BOX_VSPACE],
						   frame_on=False)
	Axes.set_axis_bgcolor('white')

	# generate boxplots
	Box = Axes.boxplot(Gradients, patch_artist=True, showfliers=False)

	# set global properties
	plt.setp(Box['medians'], color=MEDIAN, linewidth=1)
	plt.setp(Box['whiskers'], color=WHISKER, linewidth=1, linestyle='-')
	plt.setp(Box['caps'], color=WHISKER, linewidth=1)

	# modify box styling
	for i, box in enumerate(Box['boxes']):
		if np.mean(Gradients[:, i]) <= 0:
			box.set(color=BLUEEDGE, linewidth=2)
			box.set(facecolor=BLUEFACE)
		else:
			box.set(color=REDEDGE, linewidth=2)
			box.set(facecolor=REDFACE)

	# add jittered data overlays
	for i in np.arange(Gradients.shape[1]):
		plt.scatter(np.random.normal(i+1, JITTER, size=Gradients.shape[0]),
					Gradients[:, i], color=POINTS, alpha=0.2,
					marker='o', s=2, zorder=100)

	# set limits
	Axes.set_ylim(1.05 * Gradients.min(), 1.05 * Gradients.max())

	# format x axis
	if XLabel is not None:
		plt.xlabel(XLabel)
	plt.xticks(np.linspace(1, Gradients.shape[1], Gradients.shape[1]),
			   [Symbols[i] + " _" + Types[i] for i in
				np.arange(Gradients.shape[1])],
			   rotation='vertical', fontsize=BOX_FONT)
	Axes.set_xticks(np.linspace(1.5, Gradients.shape[1]-0.5,
								Gradients.shape[1]-1), minor=True)
	Axes.xaxis.set_ticks_position('bottom')

	# format y axis
	if YLabel is not None:
		plt.ylabel(YLabel)
	Axes.yaxis.set_ticks_position('left')

	# add grid lines and zero line
	Axes.xaxis.grid(True, color=GRID, linestyle='-', which='minor')
	plt.plot([0, Gradients.shape[1]+0.5], [0, 0], color='black')

	return Figure


def PairScatter(Gradients, Symbols, Types):
	"""
	Generates boxplot series of feature gradients ranked by absolute magnitude.

	Parameters:
	----------

	Gradients : array_like
	Numpy array containing feature/sample gradients obtained by RiskCohort.
	Features are in columns and samples are in rows.

	Symbols : array_like
	List containing strings describing features.

	Types: array_like
	List containing strings describing feature types (e.g. CNV, Mut, Clinical).

	Returns:
	--------
	Figure : figure handle
		Handle to figure used for saving image to disk i.e.
		Figure.savefig('heatmap.pdf')

	Notes:
	------
	Features are displayed in the order they are provided. Any sorting should
	happen prior to calling.
	"""

	# calculate means, standard deviations
	Means = np.asarray(np.mean(Gradients, axis=0))
	Std = np.asarray(np.std(Gradients, axis=0))

	# generate subplots
	Figure, Axes = plt.subplots(nrows=Gradients.shape[1],
								ncols=Gradients.shape[1],
								figsize=(PAIR_FW, PAIR_FW),
								facecolor='white')
	Figure.subplots_adjust(hspace=PAIR_SPACING, wspace=PAIR_SPACING,
						   bottom=PAIR_SPACING)

	# remove axes and ticks
	for ax in Axes.flat:
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)

	# generate scatter plots in lower triangular portion
	for i, j in zip(*np.triu_indices_from(Axes, k=1)):
		Axes[i, j].scatter((Gradients[:, j]-Means[j]) / Std[j],
						   (Gradients[:, i]-Means[i]) / Std[i],
						   color=POINTS, alpha=0.2, marker='o', s=2)
		Smooth = lowess((Gradients[:, j]-Means[j]) / Std[j],
						(Gradients[:, i]-Means[i]) / Std[i])
		Axes[i, j].plot(Smooth[:, 1], Smooth[:, 0], color='red')

	# generate histograms on diagonal
	for i in np.arange(Gradients.shape[1]):
		if Means[i] <= 0:
			Axes[i, i].hist(Gradients[:, i],
							facecolor=BLUEFACE,
							alpha=0.8)
		else:
			Axes[i, i].hist(Gradients[:, i],
							facecolor=REDFACE,
							alpha=0.8)
		Axes[i, i].annotate(Symbols[i] + " _" + Types[i], (0, 0),
							xycoords='axes fraction',
							ha='right', va='top',
							rotation=45)

	# delete unused axes
	for i, j in zip(*np.tril_indices_from(Axes, k=-1)):
		Figure.delaxes(Axes[i, j])

	return Figure


def KMPlots(Gradients, Raw, Symbols, Types, Survival, Censored):
  
    """
    Generates KM plots for individual features ranked by absolute magnitude.

    Parameters:
    ----------

    Gradients : array_like
    Numpy array containing feature/sample gradients obtained by RiskCohort.
    Features are in columns and samples are in rows.

    Raw : array_like
    Numpy array containing raw, unnormalized feature values. These are used to
    examine associations between feature values and cluster assignments.
    Features are in columns and samples are in rows.

    Symbols : array_like
    List containing strings describing features.

    Types: array_like
    List containing strings describing feature types (e.g. CNV, Mut, Clinical).
    See notes on allowed values of Types below.

    Survival : array_like
    Array containing death or last followup values.

    Censored : array_like
    Array containing vital status at last followup. 1 (alive) or 0 (deceased).

    Returns
    -------
    Figures : figure handle
    List containing handles to figures.

    Names : array_like
    List of feature names for figures in 'Figures'

    Notes
    -----
    Types like 'Mut' and 'CNV' that are generated as suffixes to feature names
    by the package tcgaintegrator are required analysis.
    Note this uses feature values as opposed to back-propagated risk gradients.
    Features are displayed in the order they are provided. Any sorting should
    happen prior to calling.
    """

    # initialize list of figures and names
    Figures = []

    # generate Kaplan Meier fitter
    kmf = KaplanMeierFitter()

    # generate KM plot for each feature
    for count, i in enumerate(np.arange(Gradients.shape[1])):

        # generate figure and axes
        Figures.append(plt.figure(figsize=(SURV_FW, SURV_FH),
                                  facecolor='white'))
        Axes = Figures[count].add_axes([SURV_HSPACE, SURV_VSPACE,
                                        1-2*SURV_HSPACE, 1-2*SURV_VSPACE])

        # initialize log-rank test result
        LogRank = None

        if Types[i] == 'Clinical':

            # get unique values to determine if binary or continuous
            Unique = np.unique(Raw[:, i])

            # process based on variable type
            if Unique.size == 2:

                # extract and plot mutant and wild-type survival profiles
                if np.sum(Raw[:, i] == Unique[0]):
                    kmf.fit(Survival[Raw[:, i] == Unique[0]],
                            1-Censored[Raw[:, i] == Unique[0]] == 1,
                            label=Symbols[i] + str(Unique[0]))
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] == Unique[1]):
                    kmf.fit(Survival[Raw[:, i] == Unique[1]],
                            1-Censored[Raw[:, i] == Unique[1]] == 1,
                            label=Symbols[i] + str(Unique[1]))
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] == Unique[0]) & \
                   np.sum(Raw[:, i] == Unique[1]):
                    LogRank = logrank_test(Survival[Raw[:, i] == Unique[0]],
                                           Survival[Raw[:, i] == Unique[1]],
                                           1-Censored[Raw[:, i] == Unique[0]]
                                             == 1,
                                           1-Censored[Raw[:, i] == Unique[1]]
                                             == 1)
                plt.ylim(0, 1)
                if LogRank is not None:
                    plt.title('Logrank p=' + str(LogRank.p_value))
                lg = plt.gca().get_legend()
                plt.setp(lg.get_texts(), fontsize=SURV_FONT)

            else:

                # determine median value
                Median = np.median(Raw[:, i])

                # extract and altered and unaltered survival profiles
                if np.sum(Raw[:, i] > Median):
                    kmf.fit(Survival[Raw[:, i] > Median],
                            1-Censored[Raw[:, i] > Median] == 1,
                            label=Symbols[i] + " > " + str(Median))
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] <= Median):
                    kmf.fit(Survival[Raw[:, i] <= Median],
                            1-Censored[Raw[:, i] <= Median] == 1,
                            label=Symbols[i] + " <= " + str(Median))
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] > Median) & np.sum(Raw[:, i] <= Median):
                    LogRank = logrank_test(Survival[Raw[:, i] > Median],
                                           Survival[Raw[:, i] <= Median],
                                           1-Censored[Raw[:, i] > Median]
                                             == 1,
                                           1-Censored[Raw[:, i] <= Median]
                                             == 1)
                plt.ylim(0, 1)
                if LogRank is not None:
                    plt.title('Logrank p=' + str(LogRank.p_value))
                lg = plt.gca().get_legend()
                plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        elif Types[i] == 'Mut':

            # extract and plot mutant and wild-type survival profiles
            if np.sum(Raw[:, i] == 1):
                kmf.fit(Survival[Raw[:, i] == 1],
                        1-Censored[Raw[:, i] == 1] == 1,
                        label=Symbols[i] + " Mutant")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] == 0):
                kmf.fit(Survival[Raw[:, i] == 0],
                        1-Censored[Raw[:, i] == 0] == 1,
                        label=Symbols[i] + " WT")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] == 1) & np.sum(Raw[:, i] == 0):
                LogRank = logrank_test(Survival[Raw[:, i] == 0],
                                       Survival[Raw[:, i] == 1],
                                       1-Censored[Raw[:, i] == 0] == 1,
                                       1-Censored[Raw[:, i] == 1] == 1)
            plt.ylim(0, 1)
            lg = plt.gca().get_legend()
            if LogRank is not None:
                plt.title('Logrank p=' + str(LogRank.p_value))
            plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        elif Types[i] == 'CNV':

            # determine if alteration is amplification or deletion
            Amplified = np.mean(Raw[:, i]) > 0

            # extract and plot altered and unaltered survival profiles
            if Amplified:
                kmf.fit(Survival[Raw[:, i] > 0],
                        1-Censored[Raw[:, i] > 0] == 1,
                        label=Symbols[i] + " " + Types[i] + " Amplified")
                kmf.plot(ax=Axes, show_censors=True)
                if(np.sum(Raw[:, i] <= 0)):
                    kmf.fit(Survival[Raw[:, i] <= 0],
                            1-Censored[Raw[:, i] <= 0] == 1,
                            label=Symbols[i] + " " + Types[i] +
                            " not Amplified")
                    kmf.plot(ax=Axes, show_censors=True)
                    LogRank = logrank_test(Survival[Raw[:, i] > 0],
                                           Survival[Raw[:, i] <= 0],
                                           1-Censored[Raw[:, i] > 0] == 1,
                                           1-Censored[Raw[:, i] <= 0] == 1)
            else:
                kmf.fit(Survival[Raw[:, i] < 0],
                        1-Censored[Raw[:, i] < 0] == 1,
                        label=Symbols[i] + " " + Types[i] + " Deleted")
                kmf.plot(ax=Axes, show_censors=True)
                if(np.sum(Raw[:, i] >= 0)):
                    kmf.fit(Survival[Raw[:, i] >= 0],
                            1-Censored[Raw[:, i] >= 0] == 1,
                            label=Symbols[i] + " " + Types[i] + " not Deleted")
                    kmf.plot(ax=Axes, show_censors=True)
                    LogRank = logrank_test(Survival[Raw[:, i] < 0],
                                           Survival[Raw[:, i] >= 0],
                                           1-Censored[Raw[:, i] < 0] == 1,
                                           1-Censored[Raw[:, i] >= 0] == 1)
            if LogRank is not None:
                plt.title('Logrank p=' + str(LogRank.p_value))
            plt.ylim(0, 1)
            lg = plt.gca().get_legend()
            plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        elif Types[i] == 'CNVArm':

            # determine if alteration is amplification or deletion
            Amplified = np.mean(Raw[:, i]) > 0

            # extract and plot altered and unaltered survival profiles
            if Amplified:
                if(np.sum(Raw[:, i] > 0.25)):
                    kmf.fit(Survival[Raw[:, i] > 0.25],
                            1-Censored[Raw[:, i] > 0.25] == 1,
                            label=Symbols[i] + " " + Types[i] + " Amplified")
                    kmf.plot(ax=Axes, show_censors=True)
                if(np.sum(Raw[:, i] <= 0.25)):
                    kmf.fit(Survival[Raw[:, i] <= 0.25],
                            1-Censored[Raw[:, i] <= 0.25] == 1,
                            label=Symbols[i] + " " + Types[i] +
                            " not Amplified")
                    kmf.plot(ax=Axes, show_censors=True)
                if(np.sum(Raw[:, i] > 0.25) & np.sum(Raw[:, i] <= 0.25)):
                    LogRank = logrank_test(Survival[Raw[:, i] > 0.25],
                                           Survival[Raw[:, i] <= 0.25],
                                           1-Censored[Raw[:, i] > 0.25] == 1,
                                           1-Censored[Raw[:, i] <= 0.25] == 1)
            else:
                if np.sum(Raw[:, i] < -0.25):
                    kmf.fit(Survival[Raw[:, i] < -0.25],
                            1-Censored[Raw[:, i] < -0.25] == 1,
                            label=Symbols[i] + " " + Types[i] + " Deleted")
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] >= -0.25):
                    kmf.fit(Survival[Raw[:, i] >= -0.25],
                            1-Censored[Raw[:, i] >= -0.25] == 1,
                            label=Symbols[i] + " " + Types[i] + " not Deleted")
                    kmf.plot(ax=Axes, show_censors=True)
                if np.sum(Raw[:, i] < -0.25) & np.sum(Raw[:, i] >= -0.25):
                    LogRank = logrank_test(Survival[Raw[:, i] < -0.25],
                                           Survival[Raw[:, i] >= -0.25],
                                           1-Censored[Raw[:, i] < -0.25] == 1,
                                           1-Censored[Raw[:, i] >= -0.25] == 1)
            plt.ylim(0, 1)
            lg = plt.gca().get_legend()
            if LogRank is not None:
                plt.title('Logrank p=' + str(LogRank.p_value))
            plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        elif (Types[i] == 'Protein') or (Types[i] == 'mRNA'):

            # determine median expression
            Median = np.median(Raw[:, i])

            # extract and altered and unaltered survival profiles
            if np.sum(Raw[:, i] > Median):
                kmf.fit(Survival[Raw[:, i] > Median],
                        1-Censored[Raw[:, i] > Median] == 1,
                        label=Symbols[i] + " " + Types[i] +
                        " Higher Expression")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] <= Median):
                kmf.fit(Survival[Raw[:, i] <= Median],
                        1-Censored[Raw[:, i] <= Median] == 1,
                        label=Symbols[i] + " " + Types[i] +
                        " Lower Expression")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] > Median) & np.sum(Raw[:, i] <= Median):
                LogRank = logrank_test(Survival[Raw[:, i] > Median],
                                       Survival[Raw[:, i] <= Median],
                                       1-Censored[Raw[:, i] > Median] == 1,
                                       1-Censored[Raw[:, i] <= Median] == 1)
            plt.ylim(0, 1)
            if LogRank is not None:
                plt.title('Logrank p=' + str(LogRank.p_value))
            lg = plt.gca().get_legend()
            plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        elif (Types[i] == 'PATHWAY'):

            # determine median expression
            Median = np.median(Raw[:, i])

            # extract and altered and unaltered survival profiles
            if np.sum(Raw[:, i] > Median):
                kmf.fit(Survival[Raw[:, i] > Median],
                        1-Censored[Raw[:, i] > Median] == 1,
                        label=Symbols[i] + " Higher Enrichment")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] <= Median):
                kmf.fit(Survival[Raw[:, i] <= Median],
                        1-Censored[Raw[:, i] <= Median] == 1,
                        label=Symbols[i] + " Lower Enrichment")
                kmf.plot(ax=Axes, show_censors=True)
            if np.sum(Raw[:, i] > Median) & np.sum(Raw[:, i] <= Median):
                LogRank = logrank_test(Survival[Raw[:, i] > Median],
                                       Survival[Raw[:, i] <= Median],
                                       1-Censored[Raw[:, i] > Median] == 1,
                                       1-Censored[Raw[:, i] <= Median] == 1)
            plt.ylim(0, 1)
            if LogRank is not None:
                plt.title('Logrank p=' + str(LogRank.p_value))
            lg = plt.gca().get_legend()
            plt.setp(lg.get_texts(), fontsize=SURV_FONT)

        else:
            raise ValueError('Unrecognized feature type ' + '"' +
                             Types[i] + '"')

    return Figures


def _SplitSymbols(Symbols):
    """
    Removes trailing and leading whitespace, separates feature types from
    feature names, enumerates duplicate symbol names
    """

    # modify duplicate symbols where needed - append index to each instance
    Prefix = [Symbol[0:str.rfind(str(Symbol), '_')] for Symbol in Symbols]
    Types = [Symbol[str.rfind(str(Symbol), '_')+1:].strip()
             for Symbol in Symbols]

    # copy prefixes
    Corrected = Prefix[:]

    # append index to each duplicate instance
    for i in np.arange(len(Prefix)):
        if Prefix.count(Prefix[i]) > 1:
            Corrected[i] = Prefix[i] + '.' + \
                str(Prefix[0:i+1].count(Prefix[i]))
        else:
            Corrected[i] = Prefix[i]

    return Corrected, Types


def _WrapSymbols(Symbols, Length=20):
    """
    Wraps long labels
    """

    # remove whitespace and wrap
    Corrected = ['\n'.join(wrap(Symbol.strip().replace('_', ' '), Length))
                 for Symbol in Symbols]

    return Corrected
