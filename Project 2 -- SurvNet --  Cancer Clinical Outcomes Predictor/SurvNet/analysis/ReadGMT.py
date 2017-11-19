def ReadGMT(File):
    """
    Reads a Gene Matrix Transposed (GMT) text file defining gene sets to
    generate lists containing gene set names, descriptions and gene set gene
    symbols.
    Parameters
    ----------
    File : string
    Filename and path to a GMT file containing gene sets.
    Returns
    -------
    GeneSets : array_like
    A list of strings containing the gene set names.
    Description : array_like
    A list of strings containing the gene set descriptions.
    Genes : array_like
    A list of lists, each containing the gene symbols for each gene set.
    Notes
    -----
    Gene sets can be obtained from the Molecular Signatures Database (MSigDB)
    at http://software.broadinstitute.org/gsea/msigdb/.
    See Also
    --------
    SSGSEA
    """

    # initialize lists for GeneSets, Links and Genes
    SetNames = []
    Descriptions = []
    Genes = []

    # open gmt file
    with open(File, 'r') as gmt:
        for Line in gmt:

            # parse line into geneset name, link and gene members
            gs, desc, gn = _ParseLine(Line)

            # append gene set to outputs
            SetNames.append(gs)
            Descriptions.append(desc)
            Genes.append(gn)

    return SetNames, Descriptions, Genes


def _ParseLine(String):
    """
    Parses a GMT file line into gene set name, descriptions and gene symbols.
    """

    # split string into delimited components
    Words = String.split()

    # extract gene set, link, genes
    GeneSet = Words[0]
    Description = Words[1]
    Genes = Words[2:]
    Genes.sort()

    return GeneSet, Description, Genes
