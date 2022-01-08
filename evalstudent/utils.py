import pandas as pd
pd.options.mode.chained_assignment = None # Disabling pandas warnings triggered by display_classes
from IPython.core.display import HTML


CLASSES_COLORS = {
    "Lead": "Grey",
    "Position": "YellowGreen",
    "Claim": "#F1C40F",
    "Counterclaim": "#E67E22",
    "Rebuttal": "#873600",
    "Evidence": "#3498DB",
    "Concluding Statement": "Green"
}


def display_classes(essay_id, train_df):
    '''
    Typically takes train_df or a model prediction as input.
    Prints the essay text (keeping its exact original formatting) using colors to highlight discourse elements and their classes.
    Uses only `predictionstring`, which is useful to display models predictions.
    '''
    
    # Handling submission format :
    discourse_type = "class" if "discourse_type" not in train_df.columns else "discourse_type"
    
    elements_df = train_df[train_df["id"] == essay_id]
    essay_text = open(f'../../raw_data/train/{essay_id}.txt').read()
    essay_words = essay_text.split()
    formatted_essay = ""
    
    # First we make sure discourse elements are in the text order
    elements_df["prediction_list"] = elements_df["predictionstring"].map(lambda x : x.split())
    elements_df["start_word_index"] = elements_df["prediction_list"].map(lambda x : int(x[0]))
    elements_df.sort_values("start_word_index", inplace=True)

    # Then for each discourse element, we go word by word trough the original essay text
    # and then we highlight the exact part of the essay corresponding to the discourse class.
    end_char = 0
    for i, element in elements_df.iterrows():
        start_word = essay_words[element["start_word_index"]] 
        start_char = essay_text[end_char:].find(start_word) + len(essay_text[:end_char])
        formatted_essay += essay_text[end_char:start_char]
        for word_index in element["prediction_list"]:
            word = essay_words[int(word_index)]
            word_position = essay_text[end_char:].find(word)
            if word_position == -1:
                return "Formatting failed"
            end_char = word_position + len(essay_text[:end_char]) + len(word)
        formatted_essay += f"|<span style='color:{CLASSES_COLORS[element[discourse_type]]}'>{essay_text[start_char:end_char]}</span>"
    formatted_essay = formatted_essay.replace("\n", "<br>")
    color_labels = " ".join([
        f"|<span style='color:{CLASSES_COLORS[class_]}'>{class_}</span>"
        for class_ in CLASSES_COLORS.keys()])
    return HTML(color_labels + "<br><br>" + formatted_essay)


def generate_predictionstring(discourse_start, discourse_end, essay_text):
    '''
    The following snippet of code was copy pasted from this Kaggle thread
    https://www.kaggle.com/c/feedback-prize-2021/discussion/297591
    It is the "official" way `predictionstring` is computed from `discourse_start/end`.
    It can be useful for models that output a prediction with character index.
    '''
    char_start = discourse_start
    char_end = discourse_end
    word_start = len(essay_text[:char_start].split())
    word_end = word_start + len(essay_text[char_start:char_end].split())
    word_end = min( word_end, len(essay_text.split()) )
    predictionstring = " ".join( [str(x) for x in range(word_start,word_end)] )
    return predictionstring