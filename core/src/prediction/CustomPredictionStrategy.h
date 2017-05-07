/*-------------------------------------------------------------------------------
  This file is part of gradient-forest.

  gradient-forest is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  gradient-forest is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with gradient-forest. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRADIENTFOREST_CUSTOMPREDICTIONSTRATEGY_H
#define GRADIENTFOREST_CUSTOMPREDICTIONSTRATEGY_H


#include "PredictionStrategy.h"

class CustomPredictionStrategy: public PredictionStrategy {
public:
  // Add more observables here as needed.
  static const std::size_t OUTCOME;

  size_t prediction_length();
  Prediction predict(size_t sampleID,
                     const std::vector<double>& averages,
                     const std::unordered_map<size_t, double>& weights_by_sampleID,
                     const Observations& observations);

  Prediction predict_with_variance(size_t sampleID,
                                   const std::vector<std::vector<size_t>>& leaf_sampleIDs,
                                   const Observations& observations,
                                   uint ci_group_size);

  bool requires_leaf_sampleIDs();
  PredictionValues precompute_prediction_values(
      const std::vector<std::vector<size_t>>& leaf_sampleIDs,
      const Observations& observations);
};


#endif //GRADIENTFOREST_CUSTOMPREDICTIONSTRATEGY_H