import numpy as np

CLASSES_LIST = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement"
]

# Code adapted from Rob Mulla (@robikscube) (https://www.kaggle.com/robikscube/student-writing-competition-twitch)
def is_match(row):
    """
    Returns `True` if prediction and ground truth are matching.
    Only used internally in get_scores.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len(set_gt)
    overlap_2 = inter / len(set_pred)
    return overlap_1 >= 0.5 and overlap_2 >= 0.5


def get_scores(pred_df, gt_df):
    """
    Returns precision, recall and f1 scores. Only used internally in kaggle_score
    for one class at a time.
    """

    # Checking DataFrames emptiness before proceeding with calculations:
    nan_metrics_nb = 0
    
    if pred_df.empty:
        precision = np.nan # Precision has no mathematical meaning in that case
        recall = 0
        nan_metrics_nb += 1
    
    if gt_df.empty:
        precision = 0
        recall = np.nan # Recall has no mathematical meaning in that case
        nan_metrics_nb += 1
    
    if nan_metrics_nb > 0:
        return {
            "precision" : precision,
            "recall" : recall,
            "f1" : np.nan if nan_metrics_nb == 2 else 0
        }
    
    # If no DataFrame is empty, we proceed:
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index

    # All ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )

    # Purposedly ignoring multiple match possibilty (very unlikely) for efficiency
    tp_df = joined[joined.apply(is_match, axis=1)]

    TP = tp_df.shape[0]
    FP = pred_df.drop(tp_df["pred_id"]).shape[0]
    FN = gt_df.drop(tp_df["gt_id"]).shape[0]
    
    # Returning metrics
    return {
        "precision" : TP / (TP + FP),
        "recall" : TP / (TP + FN),
        "f1" : TP / (TP + 0.5 * (FP + FN))
    }


def kaggle_score(pred_df, gt_df, return_details=False):
    """
    A function that scores for the kaggle Student Writing Competition

    Uses the steps in the evaluation page, with a simplified (= random)
    calculation when 2 matches exist for the same discourse element. 
    See https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    scores = [
        get_scores(
            pred_df[pred_df["class"] == class_],
            gt_df[gt_df["discourse_type"] == class_]
        )
        for class_ in CLASSES_LIST   
    ]
    f1_score = np.nanmean([class_scores["f1"] for class_scores in scores])
    if return_details:
        return f1_score, dict(zip(CLASSES_LIST, scores))
    return f1_score
