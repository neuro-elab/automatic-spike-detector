.. module:: spidet

.. _documentation:

=============
Documentation
=============

Nonnegative Matrix Factorization (NMF) is a machine-learning algorithm used to decompose an original
data matrix into two lower-rank matrices. As such it resembles other matrix decomposition algorithms,
such as Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and Independent
Component Analysis (ICA), whose purpose is to reduce the dimensionality of the original data and to
identify the underlying patterns, principal components or latent variables that make up the observed data.
The unique characteristic of NMF is that it imposes nonnegativity constraints on the values of the factorized
matrices, which only allow for additive combinations and thus result in a parts-based representation of the
original data. This characteristic improves the interpretability of the results [103], [64]. In many real-world
applications, such as image processing, text analysis or gene data analysis, the presence of negative values
in both the original data and the latent variables is physically meaningless. This limits the application
scenarios for methods like PCA and ICA [103], [38]. In [64], Lee and Seung, using image data, showed
that the decomposition of human faces performed by NMF results in a range of facial parts, such as noses,
mouths, and eyes, corresponding to the different parts making up the whole. By contrast, when applied to
the same data set, PCA yields a set of prototypical eigenfaces that by linear combination approximates the
original data.

The basic mechanism of NMF, illustrated in :num:`fig_nmf`, is that it approximates an original data matrix
by the linear combination of two nonnegative lower-rank matrices

.. math::
    V ≈W ×H

Here, :math:`V` is a :math:`M × N` matrix incorporating the original data in terms of features (rows) and samples
(columns). :math:`W` is a :math:`M × K` matrix, whose K columns are called basis vectors and represent the latent
variables, expressed by the same features as the observed data. :math:`H` is a :math:`K × N` matrix whose columns are
the encodings of the samples [64], [15].

.. _fig_nmf:

.. figure:: _images/figure_nmf.pdf

    Conceptual diagram of Nonnegative Matrix Factorization (NMF)

A further result reported by Lee and Seugn in [64] is that both the basis vectors and the encodings
contain numerous vanishing coefficients, meaning that both the basis images and encodings are sparse.
The basis images contain several versions of different facial parts that are in different locations and have
different forms, accounting for their sparsity. The sparsity in the image encodings is explained by the fact
that all parts are used by at least one image, but no image uses all parts.
In [15], Brunet et al. introduced NMF as a means of reducing the dimension of expression data from
thousands of genes to a small number of metagenes. By combining NMF with consensus clustering, a
model-selection algorithm developed in [76] and adding a quantitative evaluation to assess the robustness
of a given decomposition, they were able to identify distinct molecular patterns and cluster samples into
the different metagenes. In their model, the data matrix consists of the gene expression levels of–expressed
in terms of the variables introduced above–:math:`M` genes in :math:`N` samples. The decomposition results in a matrix
:math:`W` , consisting of :math:`K` metagenes, each expressed in terms of the :math:`M` genes, and an :math:`H` matrix
whose :math:`N` columns represent the metagene expression pattern of each individual sample. Translating NMF into the
context of EEG data while using the analogous terminology, NMF decomposes a set of EEG signals into a
small number of EEG metapatterns that make up these signals. The original data matrix :math:`V` contains the
measurements of the electrical activity in the brain from :math:`M` different electrodes along :math:`N` different time
points. The :math:`K` columns of W then represent the EEG patterns contained in the original signals, expressed
in terms of the different brain activity patterns captured by the :math:`M` electrodes, while the :math:`N`
columns of :math:`H` represent the activation level of each metapattern at each point in time.
