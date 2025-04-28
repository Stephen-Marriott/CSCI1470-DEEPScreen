import tensorflow as tf
from data_processing import get_train_test_val_data_loaders
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores
import argparse

def evaluate_model(model_path, target_id, batch_size=32):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    _, _, test_loader = get_train_test_val_data_loaders(target_id, batch_size)
    
    # Get predictions
    all_labels, all_preds, all_comp_ids = [], [], []
    for img_arrs, labels, comp_ids in test_loader:
        preds = tf.argmax(model(tf.convert_to_tensor(img_arrs, dtype=tf.float32)), axis=1).numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_comp_ids.extend(comp_ids.numpy())
    
    # Calculate metrics
    metrics = prec_rec_f1_acc_mcc(all_labels, all_preds)
    
    # Print results
    print("\nTest Performance Metrics:")
    for metric in get_list_of_scores():
        print(f"{metric}: {metrics.get(metric, 'N/A')}")
    
    # Save predictions (optional)
    with open(f"result_files/eval_{target_id}_predictions.txt", "w") as f:
        f.write("CompoundID\tLabel\tPrediction\n")
        for comp_id, label, pred in zip(all_comp_ids, all_labels, all_preds):
            f.write(f"{comp_id}\t{label}\t{pred}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--target_id", type=str, required=True, help="Target ChEMBL ID (e.g., CHEMBL286)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.target_id, args.batch_size)