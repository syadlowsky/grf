/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <math.h>
#include <cstdio>

#include "RegressionSplittingRule.h"

RegressionSplittingRule::RegressionSplittingRule(Data* data,
                                                 double alpha,
                                                 double imbalance_penalty):
    data(data),
    alpha(alpha),
    imbalance_penalty(imbalance_penalty) {
  size_t max_num_unique_values = data->get_max_num_unique_values();
  this->counter = new size_t[max_num_unique_values];
  this->sums = new double[max_num_unique_values];
  this->weights = new double[max_num_unique_values];
}

RegressionSplittingRule::~RegressionSplittingRule() {
  if (counter != 0) {
    delete[] counter;
  }
  if (sums != 0) {
    delete[] sums;
  }
  if (weights != 0) {
    delete[] weights;
  }
}

bool RegressionSplittingRule::find_best_split(size_t node,
                                              const std::vector<size_t>& possible_split_vars,
                                              const std::unordered_map<size_t, double>& labels_by_sample,
                                              const std::unordered_map<size_t, double>& weights_by_sample,
                                              const std::vector<std::vector<size_t>>& samples,
                                              std::vector<size_t>& split_vars,
                                              std::vector<double>& split_values) {

  size_t size_node = samples[node].size();
  size_t min_child_size = std::max<size_t>(std::ceil(size_node * alpha), 1uL);

  // Precompute the sum of outcomes in this node.
  double sum_node = 0.0;
  double weight_node = 0.0;
  double weight;
  for (auto& sample : samples[node]) {
    weight = weights_by_sample.at(sample);
    sum_node += weight * labels_by_sample.at(sample);
    weight_node += weight;
    // std::cout << "weight of node: " << weight_node << std::endl;
  }

  // Initialize the variables to track the best split variable.
  size_t best_var = 0;
  double best_value = 0;
  double best_decrease = -1;

  // For all possible split variables
  for (auto& var : possible_split_vars) {
    // Use faster method for both cases
    double q = (double) size_node / (double) data->get_num_unique_data_values(var);
    if (q < Q_THRESHOLD) {
      find_best_split_value_small_q(node, var, sum_node, weight_node, size_node, min_child_size,
                                    best_value, best_var, best_decrease, labels_by_sample, weights_by_sample, samples);
    } else {
      find_best_split_value_large_q(node, var, sum_node, weight_node, size_node, min_child_size,
                                    best_value, best_var, best_decrease, labels_by_sample, weights_by_sample, samples);
    }
  }

  // Stop if no good split found
  if (best_decrease < 0) {
    return true;
  }

  // Save best values
  split_vars[node] = best_var;
  split_values[node] = best_value;
  return false;
}

void RegressionSplittingRule::find_best_split_value_small_q(size_t node,
                                                            size_t var,
                                                            double sum_node,
                                                            double weight_node,
                                                            size_t size_node,
                                                            size_t min_child_size,
                                                            double& best_value, size_t& best_var,
                                                            double& best_decrease,
                                                            const std::unordered_map<size_t, double>& labels_by_sample,
                                                            const std::unordered_map<size_t, double>& weights_by_sample,
                                                            const std::vector<std::vector<size_t>>& samples) {
  std::vector<double> possible_split_values;
  data->get_all_values(possible_split_values, samples.at(node), var);

  // Try next variable if all equal for this
  if (possible_split_values.size() < 2) {
    return;
  }

  // Remove largest value because no split possible
  possible_split_values.pop_back();

  // Initialize with 0m if not in memory efficient mode, use pre-allocated space
  size_t num_splits = possible_split_values.size();
  double* sums_right;
  double* weights_right;
  size_t* n_right;
  sums_right = sums;
  n_right = counter;
  weights_right = weights;
  std::fill(sums_right, sums_right + num_splits, 0);
  std::fill(n_right, n_right + num_splits, 0);
  std::fill(weights_right, weights_right + num_splits, 0);

  // Sum in right child and possible split
  for (auto& sample : samples[node]) {
    double value = data->get(sample, var);
    double response = labels_by_sample.at(sample);
    double weight = weights_by_sample.at(sample);

    // Count samples until split_value reached
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
	weights_right[i] += weight;
        sums_right[i] += weight * response;
      } else {
        break;
      }
    }
  }

  // Compute decrease of impurity for each possible split
  for (size_t i = 0; i < num_splits; ++i) {

    // Skip this split if one child is too small.
    size_t n_left = size_node - n_right[i];
    if (n_left < min_child_size) {
      continue;
    }

    // Stop if the right child is too small.
    if (n_right[i] < min_child_size) {
      break;
    }

    double sum_right = sums_right[i];
    double sum_left = sum_node - sum_right;
    double weight_right = weights_right[i];
    double weight_left = weight_node - weight_right;
    // std::cout << "size: " << size_node << std::endl;
    // std::cout << "weight left: " << weight_left
    //           << "n left: " << n_left
    //           << "weight right: " << weight_right
    //           << "n right: " << n_right[i]
    //           << std::endl;
    double decrease = sum_left * sum_left / weight_left + sum_right * sum_right / weight_right;

    // Penalize splits that are too close to the edges of the data.
    double penalty = imbalance_penalty * (1.0 / n_left + 1.0 / n_right[i]);
    decrease -= penalty;


    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = possible_split_values[i];
      best_var = var;
      best_decrease = decrease;
    }
  }
}

void RegressionSplittingRule::find_best_split_value_large_q(size_t node,
                                                            size_t var,
                                                            double sum_node,
                                                            double weight_node,
                                                            size_t size_node,
                                                            size_t min_child_size,
                                                            double& best_value,
                                                            size_t& best_var,
                                                            double& best_decrease,
                                                            const std::unordered_map<size_t, double>& responses_by_sample,
                                                            const std::unordered_map<size_t, double>& weights_by_sample,
                                                            const std::vector<std::vector<size_t>>& samples) {
  // Set counters to 0
  size_t num_unique = data->get_num_unique_data_values(var);
  std::fill(counter, counter + num_unique, 0);
  std::fill(weights, weights + num_unique, 0);
  std::fill(sums, sums + num_unique, 0);

  double weight;
  for (auto& sample : samples[node]) {
    size_t index = data->get_index(sample, var);

    weight = weights_by_sample.at(sample);
    sums[index] += weight * responses_by_sample.at(sample);
    weights[index] += weight;
    ++counter[index];
  }

  size_t n_left = 0;
  double sum_left = 0;
  double weight_left = 0;

  // Compute decrease of impurity for each split
  for (size_t i = 0; i < num_unique - 1; ++i) {
    n_left += counter[i];
    sum_left += sums[i];
    weight_left += weights[i];

    // Skip to the next value if the left child is too small.
    if (n_left < min_child_size) {
      continue;
    }

    // Stop if the right child is too small.
    size_t n_right = size_node - n_left;
    if (n_right < min_child_size) {
      break;
    }

    double sum_right = sum_node - sum_left;
    double weight_right = weight_node - weight_left;
    // std::cout << "weight node: " << weight_node << " ";
    // std::cout << "Q weight left: " << weight_left
    //           << " n left: " << n_left
    //           << " weight right: " << weight_right
    //           << " n right: " << n_right
    //           << std::endl;
    double decrease = sum_left * sum_left / weight_left + sum_right * sum_right / weight_right;

    // Penalize splits that are too close to the edges of the data.
    double penalty = imbalance_penalty * (1.0 / n_left + 1.0 / n_right);
    decrease -= penalty;

    // If better than before, use this
    if (decrease > best_decrease) {
      best_value = data->get_unique_data_value(var, i);
      best_var = var;
      best_decrease = decrease;
    }
  }
}
