# Copyright (C) Tahoe Therapeutics 2024-2025. All rights reserved.
import anndata
from scipy import sparse


## From https://github.com/scverse/scvi-tools/issues/3018  ## 

class ArrayFakerSparse(sparse.csr_matrix):
    """Create a wrapper around a sparse matrix layer in an AnnCollection that
    obeys the typical AnnData API"""

    def __init__(self, collection, key):
        self.collection = collection
        self.key = key
        # minibatch size for batchwise operations
        self.batch_size = 256

    def __getitem__(self, idx):
        """Second layer of get item that handles returning an index
        into a specific layer"""
        return self.collection[idx].layers[self.key]

    def getnnz(self):
        """Run through the data in minibatches and count non-zero elements"""
        n_batches = int(math.ceil(self.collection.shape[0] / n_batches))

        nnz = 0

        start = 0
        end = self.batch_size
        for _i in range(n_batches):
            batch = self.collection[start:end].layers[self.key]
            nnz += batch.getnnz()
            start = end
            end = min(end + self.batch_size, self.collection.shape[0])

        return nnz

    def _mul_vector(self, other):
        """Included to try and trick issparse"""
        raise NotImplementedError("This should basically never be called.")
        return self._data.dot(other)

    def _mul_multivector(self, other):
        """Included to try and trick issparse"""
        raise NotImplementedError("This should basically never be called.")
        return self._data.dot(other)

    def toarray(self,):
        """Convert to a dense array.

        Notes
        -----
        Ths would require full dense representation in memory. It shouldn't be
        used.
        """
        raise NotImplementedError("This should basically never be called.")
        return self.collection[:].layers[self.key].toarray()

    def transpose(self,):
        """Transpose the entire array.

        Notes
        -----
        Ths would require full dense representation in memory. It shouldn't be
        used.
        """
        raise NotImplementedError("This should basically never be called.")
        return self.collection[:].layers[self.key].transpose()

    def __repr__(self):
        return f"ArrayFakerSparse(collection={self.collection}, key={self.key})"

    @property
    def data(self):
        """Implement a reference into a subset of the non-zero values.

        Notes
        -----
        This is used in `scvi-tools` to check that values are integers. We
        just return a subsample for now so that existing code paths are
        supported.
        """
        return self.collection[:64].layers[self.key].data

    def getformat(self):
        """Extract the relevant format (CSR, CSC, COO)"""
        return self.collection[:4].layers[self.key].getformat()


class LayerFaker:
    """Custom access for .layers inside an AnnCollection that lives
    within a CollectionAdapter"""

    def __init__(self, collection):
        self.collection = collection

    def keys(self):
        k = self.collection[1:5].layers.keys()
        return k

    def __getitem__(self, key):
        """First layer of get item that handles returning an index into a specific layer"""
        # determine the array's type
        # we first need to draw an AnnCollectionView to get an object that has
        # access to .layers, which is not provided in AnnCollection
        sample = self.collection[:4]
        layer = sample.layers[key]

        if isinstance(layer, sparse.spmatrix):
            return ArrayFakerSparse(self.collection, key)
        elif isinstance(layer, np.ndarray):
            return ArrayFakerNDarray(self.collection, key)
        else:
            raise TypeError(f"Unknown array type {type(layer)}")


class CollectionAdapter:
    """Allow an AnnCollection to pretend to be an AnnData."""
    # scvi-tools stores registry information here
    uns = {}
    # necessary for an scvi-tools check
    isbacked = True
    # necessary to pass an scvi-tools check that rejects views
    is_view = False

    SPECIAL_ATTRS = [
        "uns",
        "is_view",
        "isbacked",
        "layers",
    ]

    def __init__(self, collection):
        self.collection = collection

    def __getattr__(self, name):
        if name in self.SPECIAL_ATTRS:
            return getattr(self, name)
        else:
            return getattr(self.collection, name)

    def __repr__(self):
        return "Adapter for:\n" + repr(self.collection)

    @property
    def layers(self):
        return LayerFaker(self.collection)

    def __getitem__(self, idx):
        return self.collection[idx]

    def __len__(self):
        return len(self.collection)

    @property
    def __class__(self):
        return anndata.AnnData


## ##

