import mlflow
from mlflow.recipes import Recipe
import mlflow

r = Recipe(profile="local")

# Clean the recipe
r.clean()

# Inspect the recipe
# r.inspect()

# Run the 'ingest' step
r.run("ingest")
