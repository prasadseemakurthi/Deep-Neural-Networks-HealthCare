def WriteGCT(Genes, Samples, Scores, File):
    """
    Writes a gene expression (GCT) file format defining the scores of features
    for individual samples.

    Parameters
    ----------
    Genes : array_like
    An N-length list of gene symbols associated with the rows of 'Scores'.

    Samples : array_like
    A K-length list of sample identifiers associated with the columns of
    'Scores'. If None samples will be enumerated.

    Scores : array_like
    An NxK-length numpy array containing the signed values associated with
    'Genes' and Samples.

    File : string
    Filename and path to write the output .gct file to.

    Notes
    -----
    This is typically used to form the input for a SSGSEA analysis. See
    http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php for more
    details on file formats.

    See Also
    --------
    FeatureAnalysis
    """

    # open rnk file
    try:
        Gct = open(File, 'w')
    except IOError:
        print "Cannot create file ", File

    # write leading rows
    Gct.write('#1.2\n')
    Gct.write(str(Scores.shape[1]) + '\t' + str(Scores.shape[0]) + '\n')
    Gct.write("NAME\tDescription\t")
    if Samples is None:
        for i in range(Scores.shape[0]-1):
            Gct.write("Sample." + str(i+1) + '\t')
        Gct.write("Sample." + str(Scores.shape[0]) + '\n')
    else:
        for i, Sample in enumerate(Samples):
            if i < len(Samples)-1:
                Gct.write(Sample + '\t')
            else:
                Gct.write(Sample + '\n')

    # write contents to file
    for i, Symbol in enumerate(Genes):
        Gct.write(Symbol + '\t\t')
        for j in range(Scores.shape[0]-1):
            Gct.write(str(Scores[j, i]) + '\t')
        Gct.write(str(Scores[-1, i]) + '\n')

    # close file
    Gct.close()
