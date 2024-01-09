#! /usr/bin/env python

from mlflow.recipes import Recipe

from myf.csv_file_format import CSVFileFormat, FeatureDtype

recipe = Recipe(profile="local")
recipe.run()
