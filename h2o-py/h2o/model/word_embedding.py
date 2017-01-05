# -*- encoding: utf-8 -*-
"""
Word embedding model.

:copyright: (c) 2017 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from h2o.utils.compatibility import *  # NOQA

from .model_base import ModelBase
from .metrics_base import *
import h2o


class H2OWordEmbeddingModel(ModelBase):
    def find_synonyms(self, word, count=20):
        """Reconstruct the training data from the GLRM model and impute all missing
        values.

        Parameters
        ----------
          test_data : H2OFrame
            The dataset upon which the H2O GLRM model was trained.

          reverse_transform : logical
            Whether the transformation of the training data during model-building should
            be reversed on the reconstructed frame.

        Returns
        -------
          Return the approximate reconstruction of the training data.
        """
        # GET /3/Word2VecSynonyms, parms: {model=Word2Vec_model_python_1483647082553_1, count=20, word=horse}
        j = h2o.api("GET /3/Word2VecSynonyms?model=%s&count=%s&word=%s" % (self.model_id, count, word))
        return j
