import numpy as np

# Code adapted from Rob Mulla (@robikscube) (https://www.kaggle.com/robikscube/student-writing-competition-twitch)
# Can most probably be optimized (if we have performance issues).

CLASSES_LIST = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement"
]


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives. Only used internally in get_scores.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return overlap_1 >= 0.5 and overlap_2 >= 0.5


def get_scores(pred_df, gt_df):
    """
    Deprecated. Returns precision, recall and f1 scores. Only used internally in kaggle_score
    for one class at a time.
    """

    # Checking DataFrames emptiness before proceeding with calculations:
    if gt_df.empty and pred_df.empty:
        return {
        "precision" : np.nan,
        "recall" : np.nan,
        "f1" : np.nan
    }
    if gt_df.empty:
        return {
            "precision" : 0,
            "recall" : np.nan, # Recall has no mathematical meaning in that case
            "f1" : 0
        }
    
    if pred_df.empty:
        return {
            "precision" : np.nan, # Precision has no mathematical meaning in that case
            "recall" : 0,
            "f1" : 0
        }
    
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["potential_TP"] = joined.apply(calc_overlap, axis=1)

    tp_df = joined[joined["potential_TP"] == True] # Purposedly ignoring multiple match possibilty (very unlikely) for efficiency
    TP = tp_df.shape[0]
    FP = pred_df.drop(tp_df["pred_id"]).shape[0]
    FN = gt_df.drop(tp_df["gt_id"]).shape[0]
    
    # return metrics
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
    detailed_scores = {}
    for class_ in CLASSES_LIST:
        pred_subset = pred_df.loc[pred_df["class"] == class_].reset_index(drop=True).copy()
        gt_subset = gt_df.loc[gt_df["discourse_type"] == class_].reset_index(drop=True).copy()
        class_scores = get_scores(pred_subset, gt_subset)
        detailed_scores[class_] = class_scores
    f1_score = np.nanmean([class_scores["f1"] for class_scores in detailed_scores.values()])
    if return_details:
        return f1_score, detailed_scores
    return f1_score


# Old version of get_scores, which had perf issues. Used a different version of calc_overlap
def get_scores_slow(pred_df, gt_df):
    """
    Broken. Returns precision, recall and f1 scores. Only used internally in kaggle_score
    for one class at a time.
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    
    # The 2 following lines can be avoided if calc_overlap returns directly the proper data type
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    
    # return metrics
    return {
        "precision" : TP / (TP + FP),
        "recall" : TP / (TP + FN),
        "f1" : TP / (TP + 0.5 * (FP + FN))
    }
