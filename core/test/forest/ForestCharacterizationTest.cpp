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

#include "PredictionStrategy.h"
#include "utility.h"
#include "ForestPredictor.h"
#include "ForestTrainer.h"
#include "FileTestUtilities.h"
#include "ForestTestUtilities.h"

#include "ForestTrainers.h"
#include "ForestPredictors.h"

#include "catch.hpp"

bool equal_predictions(const std::vector<Prediction>& actual_predictions,
                       const std::vector<std::vector<double>>& expected_predictions) {
  if (actual_predictions.size() != expected_predictions.size()) {
    return false;
  }
  for (size_t i = 0; i < actual_predictions.size(); i++) {
    Prediction prediction = actual_predictions[i];
    std::vector<double> expected_prediction = expected_predictions[i];
    if (prediction.size() != expected_prediction.size()) {
      return false;
    }

    for (size_t j = 0; j < prediction.size(); j++) {
      double value = prediction.get_predictions()[j];
      if (!equalDoubles(value, expected_prediction[j], 1e-2)) {
        return false;
      }
    }
  }
  return true;
}

void update_predictions_file(const std::string file_name, std::vector<Prediction> predictions) {
  std::vector<std::vector<double>> values;
  for (auto& prediction : predictions) {
    values.push_back(prediction.get_predictions());
  }
  FileTestUtilities::writeCsvFile(file_name, values);
}

TEST_CASE("quantile forest predictions have not changed", "[quantile], [characterization]") {
  std::vector<double> quantiles({0.25, 0.5, 0.75});
  Data *data = loadDataFromFile("test/forest/resources/quantile_data.csv");

  ForestTrainer trainer = ForestTrainers::quantile_trainer(data, 10, quantiles);
  ForestTestUtilities::init_trainer(trainer);
  Forest forest = trainer.train(data);

  ForestPredictor predictor = ForestPredictors::quantile_predictor(4, quantiles);

  std::vector<Prediction> oob_predictions = predictor.predict_oob(forest, data);
  std::vector<std::vector<double>> expected_oob_predictions = FileTestUtilities::readCsvFile(
      "test/forest/resources/quantile_oob_predictions.csv");
  REQUIRE(equal_predictions(oob_predictions, expected_oob_predictions));

  std::vector<Prediction> predictions = predictor.predict(forest, data);
  std::vector<std::vector<double>> expected_predictions = FileTestUtilities::readCsvFile(
      "test/forest/resources/quantile_predictions.csv");
  REQUIRE(equal_predictions(predictions, expected_predictions));

  delete data;
}

TEST_CASE("causal forest predictions have not changed", "[causal], [characterization]") {
  Data* data = loadDataFromFile("test/forest/resources/causal_data.csv");

  ForestTrainer trainer = ForestTrainers::instrumental_trainer(data, 10, 11, 11, 0);
  ForestTestUtilities::init_trainer(trainer);

  Forest forest = trainer.train(data);

  ForestPredictor predictor = ForestPredictors::instrumental_predictor(4, 1);

  std::vector<Prediction> oob_predictions = predictor.predict_oob(forest, data);
  std::vector<std::vector<double>> expected_oob_predictions = FileTestUtilities::readCsvFile(
      "test/forest/resources/causal_oob_predictions.csv");
  REQUIRE(equal_predictions(oob_predictions, expected_oob_predictions));

  std::vector<Prediction> predictions = predictor.predict(forest, data);
  std::vector<std::vector<double>> expected_predictions = FileTestUtilities::readCsvFile(
      "test/forest/resources/causal_predictions.csv");
  REQUIRE(equal_predictions(predictions, expected_predictions));

  delete data;
}

TEST_CASE("regression forest predictions have not changed", "[regression], [characterization]") {
  Data* data = loadDataFromFile("test/forest/resources/regression_data.csv");

  ForestTrainer trainer = ForestTrainers::regression_trainer(data, 10);
  ForestTestUtilities::init_trainer(trainer);

  Forest forest = trainer.train(data);

  ForestPredictor predictor = ForestPredictors::regression_predictor(4);

  std::vector<Prediction> oob_predictions = predictor.predict_oob(forest, data);
  std::vector<std::vector<double>> expected_oob_predictions = FileTestUtilities::readCsvFile(
      "test/forest/resources/regression_oob_predictions.csv");
  REQUIRE(equal_predictions(oob_predictions, expected_oob_predictions));

  std::vector<Prediction> predictions = predictor.predict(forest, data);
  std::vector<std::vector<double>> expected_predictions = FileTestUtilities::readCsvFile(
  "test/forest/resources/regression_predictions.csv");
  REQUIRE(equal_predictions(predictions, expected_predictions));

  delete data;
}
