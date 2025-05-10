import logging
import os
import mlflow.sklearn
from dotenv import load_dotenv
import mlflow
from mlflow.models import infer_signature
import dagshub  
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI.
    """
    # Use a default URI if MLFLOW_TRACKING_URI is not set
    uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/ajitkumarsenthil5/citibike_prediciton_aml")
    mlflow.set_tracking_uri(uri)
    logger.info(f"MLflow tracking URI set to: {uri}")

    return mlflow

# Initialize logger (this is redundant since it's already defined above, but keeping as per original code)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="metric",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log a trained model, parameters, and metrics to MLflow.

    Parameters:
    - model: Trained model object (e.g., sklearn model).
    - input_data: Input data used for training (for signature inference).
    - experiment_name: Name of the MLflow experiment.
    - metric_name: Name of the metric to log (e.g., "RMSE", "accuracy").
    - model_name: Optional name for the registered model.
    - params: Optional dictionary of hyperparameters to log.
    - score: Optional evaluation metric to log.
    """
    try:
        # Load environment variables
        load_dotenv()

        # Initialize Dagshub with MLflow integration
        dagshub.init(repo_owner='ajitkumarsenthil5', repo_name='citibike_prediciton_aml', mlflow=True)
        logger.info("Dagshub initialized with MLflow integration.")

        # Set tracking URI from environment or fallback (Dagshub.init should set this, but we keep a fallback)
        uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/ajitkumarsenthil5/citibike_prediciton_aml")
        mlflow.set_tracking_uri(uri)
        logger.info(f"Using MLflow tracking URI: {uri}")

        # Set experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
                logger.info(f"Logged parameters: {params}")

            # Log score/metric
            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"Logged {metric_name}: {score}")

            # Infer signature
            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("Model signature inferred.")

            # Set default model name if not provided
            if not model_name:
                model_name = model.__class__.__name__

            # Log model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )
            logger.info(f"Model logged and registered as: {model_name}")

            return model_info

    except Exception as e:
        logger.error(f"An error occurred while logging to MLflow: {e}")
        raise