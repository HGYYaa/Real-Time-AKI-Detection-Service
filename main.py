import time
import csv
import logging
from src import data
from src import inference
from src import network

# Path to the ground truth CSV file for performance evaluation
GROUND_TRUTH_PATH = "data/aki.csv"

# Configure logging to output INFO level messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory storage for positive AKI predictions to be used in run_evaluation
_system_alerts = []


def receive_HL7(raw_HL7: str) -> None:
    """
    Callback function triggered when an HL7 message is received.
    Handles parsing, inference, alerting, and data persistence.
    """
    try:
        # 1. Data Processing: Parse raw HL7 string into structured data
        result = data.process_HL7(raw_HL7)
        if result is None: return
        mrn, history, gender = result

        # 2. Inference: Check for Acute Kidney Injury (AKI) based on gender and history
        is_aki = inference.predict(gender, history)

        # 3. Alerting Logic: If AKI is detected
        if is_aki:
            # Extract timestamp (formatted as string) from the latest history entry
            latest_timestamp = str(history[-1][0])[:12]

            # Record the alert for post-run evaluation
            _system_alerts.append((str(mrn), latest_timestamp))

            # Send the alert via the network module
            network.alert(mrn, latest_timestamp)

        # 4. Persistence: Save updated patient history to disk
        data.save_patient_history(mrn)

    except Exception:
        logger.exception("------ Failed to process HL7 message ------")


def run_evaluation():
    """
    Calculates performance metrics (F3 Score) by comparing system alerts
    against the ground truth CSV file. Handles missing file gracefully.
    """
    try:
        # Load Ground Truth
        with open(GROUND_TRUTH_PATH, 'r') as f:
            # Read CSV and normalize date format: '2024-03-31 22:09:00' -> '202403312209'
            # Using set comprehension for efficiency
            gt = {(r['mrn'], r['date'].replace("-", "").replace(":", "").replace(" ", "")[:12]) for r in csv.DictReader(f)}

        # Convert recorded system alerts to a set for comparison
        pred = set(_system_alerts)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)

        # Calculate Precision and Recall (handle division by zero)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0

        # Calculate F3 Score (Beta = 3, weighting Recall higher than Precision)
        f3 = (10 * p * r) / (9 * p + r) if (p + r) else 0.0

        print(f"[Eval] TP:{tp} FP:{fp} FN:{fn} | Precision:{p:.4f} Recall:{r:.4f} F3 Score:{f3:.4f}")

    except FileNotFoundError:
        # Gracefully handle the case where the ground truth file is missing (replaces os.path.exists)
        logger.warning(f"Evaluation skipped: '{GROUND_TRUTH_PATH}' not found.")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def main():
    """
    Main entry point: Initializes subsystems and starts the network listener.
    """
    # Initialize Data module (database connections/cache)
    data.initialize()
    logger.info(">>>>>> data.initialize() <<<<<<")

    # Initialize Inference module (load models)
    inference.initialize()
    logger.info(">>>>>> inference.initialize() <<<<<<")

    # Initialize Network module and register the callback function
    network.initialize(pass_HL7=receive_HL7)
    logger.info(">>>>>> network.initialize() <<<<<<")

    try:
        # Keep the main thread alive to listen for incoming messages
        network.wait()
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        network.stop()
        network.wait()

    # Run performance evaluation (file check is handled inside the function now)
    run_evaluation()

    logger.info(">>>>>> System closed <<<<<<")


if __name__ == "__main__":
    main()
