import datasets
import torchmetrics
import torchmetrics.retrieval

_DESCRIPTION = """
This metric calculates various information retrieval metrics. It uses torchmetrics for calculating the metric.
Available metrics are 
- HitRate
- FallOut
- Mean Average Precision (MAP)
- Mean Reciprical Rank (MRR)
- Normalized DCG (nDCG)
- Retrieval Precision
- Retrieval Recall
"""

_CITATION="""
"""

_KWARGS_DESCRIPTION = """
Computes the metric given as config.
Args:
    predictions : 
    references :
    indices :
    
Returns:
    The metric value
    
Examples:

 >>> predictions = [0.8, -0.4, 1.0, 1.4, 0.0]
 >>> references = [0, 1, 0, 1, 1]
 >>> indices = [0, 0, 1, 1, 1]
 >>> ir_metric = datasets.load_metric("ir")
 >>> results = ir_metric.compute(predictions=predictions, references=references, indices=indices)
 >>> print(results)
 {'hitrate': ,
  'fallout': ,
  'map': ,
  'mrr': ,
  'ndcg': ,
  'rPrec': ,
  'rRecl': }
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RetrievalMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float16")),
                    "references": datasets.Sequence(datasets.Value("uint16")),
                    "indices": datasets.Sequence(datasets.Value("uint16")),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _get_metric_fn(self,metric):
        
        if metric=='hitrate':
            return torchmetrics.RetrievalHitRate()
        elif metric=='fallout':
            return torchmetrics.RetrievalFallOut()
        elif metric=='map':
            return torchmetrics.RetrievalMAP()
        elif metric=='mrr':
            return torchmetrics.RetrievalMRR()
        elif metric=='ndcg':
            return torchmetrics.RetrievalNormalizedDCG()
        elif metric=='rprecision':
            return torchmetrics.RetrievalPrecision()
        elif metric=='rrecall':
            return torchmetrics.RetrievalRecall()
        
    def _compute(self, predictions, references, indices, metric_list=None):
        
        all_metrics=['hitrate','fallout','map','mrr','ndcg','rprecision','rrecall']
        
        score = {}
        for metric in metric_list:
            if metric not in all_metrics:
                print(f"Error! given metric name {metric} not found in available metrics. Skipping..")
                continue
            score[metric] = self._get_metric_fn(metric)(predictions, references, indexes=indices)
        
        return score