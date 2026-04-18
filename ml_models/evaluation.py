"""
Model evaluation and training pipeline for recommender system.
Handles evaluation, fine-tuning, and performance monitoring.
"""
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance on recommendation tasks."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}
    
    def evaluate_retrieval(
        self,
        embeddings_db: np.ndarray,
        query_embeddings: np.ndarray,
        ground_truth: Dict[int, List[int]],
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance using standard metrics.
        
        Args:
            embeddings_db: Database embeddings (N, D)
            query_embeddings: Query embeddings (Q, D)
            ground_truth: Ground truth relevance (query_idx -> [relevant_db_indices])
            k_values: K values for metrics (default [1, 5, 10])
            
        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [1, 5, 10]
        
        try:
            from sklearn.metrics import ndcg_score
            
            metrics = {}
            
            # Normalize embeddings
            embeddings_db = embeddings_db / (np.linalg.norm(embeddings_db, axis=1, keepdims=True) + 1e-8)
            query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Compute similarity matrix
            similarity = query_embeddings @ embeddings_db.T
            
            # Calculate metrics for each k
            for k in k_values:
                top_k_indices = np.argsort(-similarity, axis=1)[:, :k]
                
                # Recall@K
                recall = self._calculate_recall_at_k(top_k_indices, ground_truth, k)
                metrics[f"recall@{k}"] = recall
                
                # Precision@K
                precision = self._calculate_precision_at_k(top_k_indices, ground_truth, k)
                metrics[f"precision@{k}"] = precision
                
                # MRR@K
                mrr = self._calculate_mrr_at_k(top_k_indices, ground_truth, k)
                metrics[f"mrr@{k}"] = mrr
                
                # NDCG@K
                ndcg = self._calculate_ndcg_at_k(similarity, ground_truth, k)
                metrics[f"ndcg@{k}"] = ndcg
            
            # Aggregate metrics
            metrics["avg_recall"] = float(np.mean([v for key, v in metrics.items() if "recall" in key]))
            metrics["avg_precision"] = float(np.mean([v for key, v in metrics.items() if "precision" in key]))
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {e}")
            return {}
    
    def _calculate_recall_at_k(self, predictions: np.ndarray, ground_truth: Dict, k: int) -> float:
        """Calculate Recall@K."""
        recalls = []
        for query_idx, top_k in enumerate(predictions):
            if query_idx in ground_truth:
                relevant = set(ground_truth[query_idx])
                retrieved = set(top_k)
                if len(relevant) > 0:
                    recall = len(relevant & retrieved) / len(relevant)
                    recalls.append(recall)
        return float(np.mean(recalls)) if recalls else 0.0
    
    def _calculate_precision_at_k(self, predictions: np.ndarray, ground_truth: Dict, k: int) -> float:
        """Calculate Precision@K."""
        precisions = []
        for query_idx, top_k in enumerate(predictions):
            if query_idx in ground_truth:
                relevant = set(ground_truth[query_idx])
                retrieved = set(top_k)
                precision = len(relevant & retrieved) / k if k > 0 else 0
                precisions.append(precision)
        return float(np.mean(precisions)) if precisions else 0.0
    
    def _calculate_mrr_at_k(self, predictions: np.ndarray, ground_truth: Dict, k: int) -> float:
        """Calculate Mean Reciprocal Rank@K."""
        mrrs = []
        for query_idx, top_k in enumerate(predictions):
            if query_idx in ground_truth:
                relevant = set(ground_truth[query_idx])
                for rank, item in enumerate(top_k[:k], 1):
                    if item in relevant:
                        mrrs.append(1.0 / rank)
                        break
                else:
                    mrrs.append(0.0)
        return float(np.mean(mrrs)) if mrrs else 0.0
    
    def _calculate_ndcg_at_k(self, relevance_scores: np.ndarray, ground_truth: Dict, k: int) -> float:
        """Calculate NDCG@K."""
        try:
            from sklearn.metrics import ndcg_score
            
            ndcgs = []
            for query_idx in range(len(relevance_scores)):
                if query_idx in ground_truth:
                    # Create binary relevance vector
                    relevant_items = set(ground_truth[query_idx])
                    y_true = np.array([1 if i in relevant_items else 0 for i in range(len(relevance_scores[query_idx]))])
                    y_score = relevance_scores[query_idx]
                    
                    ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
                    ndcgs.append(ndcg)
            
            return float(np.mean(ndcgs)) if ndcgs else 0.0
        except Exception as e:
            logger.warning(f"Error calculating NDCG: {e}")
            return 0.0


class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, metrics_file: str = "logs/model_metrics.json"):
        """Initialize model monitor."""
        self.metrics_file = Path(metrics_file)
        self.metrics = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
    
    def _save_metrics(self):
        """Save metrics to disk."""
        self.metrics_file.parent.mkdir(exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_inference(
        self,
        query_id: str,
        inference_time_ms: float,
        results_count: int,
        top_similarity: float
    ):
        """Log inference metrics."""
        timestamp = datetime.utcnow().isoformat()
        
        self.metrics[timestamp] = {
            "query_id": query_id,
            "inference_time_ms": inference_time_ms,
            "results_count": results_count,
            "top_similarity": top_similarity,
        }
        
        self._save_metrics()
    
    def get_performance_stats(self, window_size: int = 100) -> Dict[str, float]:
        """Get performance statistics from recent inferences."""
        if not self.metrics:
            return {}
        
        recent = list(self.metrics.values())[-window_size:]
        
        inference_times = [m["inference_time_ms"] for m in recent]
        similarities = [m["top_similarity"] for m in recent if m.get("top_similarity")]
        
        return {
            "avg_inference_time_ms": float(np.mean(inference_times)),
            "p95_inference_time_ms": float(np.percentile(inference_times, 95)),
            "p99_inference_time_ms": float(np.percentile(inference_times, 99)),
            "avg_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "min_similarity": float(np.min(similarities)) if similarities else 0.0,
            "max_similarity": float(np.max(similarities)) if similarities else 0.0,
        }
    
    def detect_performance_drift(self, baseline_stats: Dict[str, float] = None, threshold: float = 0.2) -> bool:
        """
        Detect if model performance drifts from baseline.
        
        Args:
            baseline_stats: Baseline performance statistics
            threshold: Drift threshold (percentage change)
            
        Returns:
            True if drift detected
        """
        current_stats = self.get_performance_stats()
        
        if not baseline_stats or not current_stats:
            return False
        
        # Check inference time drift
        if "avg_inference_time_ms" in baseline_stats and "avg_inference_time_ms" in current_stats:
            baseline_time = baseline_stats["avg_inference_time_ms"]
            current_time = current_stats["avg_inference_time_ms"]
            
            drift = abs(current_time - baseline_time) / (baseline_time + 1e-8)
            if drift > threshold:
                logger.warning(f"Performance drift detected: {drift*100:.2f}% increase in inference time")
                return True
        
        return False


class TrainingPipeline:
    """Training and fine-tuning pipeline for models."""
    
    def __init__(self):
        """Initialize training pipeline."""
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "metrics": [],
        }
    
    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        device: str = "cuda",
        log_interval: int = 100
    ) -> float:
        """Train for one epoch."""
        try:
            import torch
            
            model.train()
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % log_interval == 0:
                    logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return float('inf')
    
    def evaluate(self, model, val_loader, device: str = "cuda") -> float:
        """Evaluate model on validation set."""
        try:
            import torch
            
            model.eval()
            total_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(val_loader)
            self.history["val_loss"].append(avg_loss)
            return avg_loss
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return float('inf')
