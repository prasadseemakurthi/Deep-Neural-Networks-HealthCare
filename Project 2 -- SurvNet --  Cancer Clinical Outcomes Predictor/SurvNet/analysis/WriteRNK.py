import numpy as np


def WriteRNK(Genes, Scores, File):
    """
    Writes a ranked-list (RNK) file format defining the ranks of features.
    Features are sorted based on the signed values provided in 'Scores'.

    Parameters
    ----------
    Genes : array_like
    An N-length list of gene symbols associated with 'Scores'.

    Scores : array_like
    An N-length numpy array containing the signed values associated with
    'Genes'.

    File : string
    Filename and path to write the output .rnk file to.

    Notes
    -----
    This is typically used to form the input for a GSEAPreranked analysis. See
    http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php for more
    details on file formats.

    See Also
    --------
    FeatureAnalysis
    """

    # sort inputs by signed score
    Order = np.argsort(Scores)

    # open rnk file
    try:
        Rnk = open(File, 'w')
    except IOError:
        print "Cannot create file ", File

    # write contents to file
    for i in Order:
        Rnk.write(Genes[i] + '\t' + str(Scores[i]) + '\n')

    # close file
    Rnk.close()
