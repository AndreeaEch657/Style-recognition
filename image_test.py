import logging
from ludwig.api import LudwigModel
#from ludwig.visualize import confusion_matrix
from pandas import read_csv
import pandas as pd

train_df_1 = read_csv("impressionism.csv")
train_df_2 = read_csv("post-impressionism.csv")
train_df_3 = read_csv("northern-renaissance.csv")


labels = ["Impressionism", "Post_Impressionism", "Northern_Renaissance"]
train_stats = {}
preprocessed_data = {}
output_directory = {}
# Constructs Ludwig model from config dictionary
model = LudwigModel(config='config.yaml', logging_level=logging.DEBUG)
train_df = {'Impressionism': train_df_1, 'Post_Impressionism': train_df_2, 'Northern_Renaissance': train_df_3}

for style in labels:
    temp = labels.copy()
    temp.remove(style)
    train_df_1 = train_df[style]
    train_df_2 = train_df[temp[0]]
    #train_df_2 = train_df[style].replace(temp[0], "Other")
    train_df_3 = train_df[temp[1]]
    #train_df_3 = train_df[style].replace(temp[1], "Other")
    min_images = 352
    train_df_1 = train_df_1.sample(n = min_images)
    train_df_2, train_df_3 = train_df_2.sample(n = int(min_images/2)), train_df_3.sample(n = int(min_images/2))
    dataframes = [train_df_1, train_df_2, train_df_3]
    train_df_2 = dataframes[1].replace(temp[0], "Other")
    train_df_3 = dataframes[2].replace(temp[1], "Other")
    
    #print(train_df_1.head())
    #print("---------------------------u2-------------------------------------")
    #print(train_df_2.head())
    #print("---------------------------u3-------------------------------------")
    #print(train_df_3.head())
    
    result = pd.concat([train_df_1, train_df_2, train_df_3])

    # Trains the model. This cell might take a few minutes.
    train_stats[style], preprocessed_data[style], output_directory[style] = model.train(dataset=result,experiment_name=style)
   

# # create Ludwig configuration dictionary
# config = {
#   'input_features': [
#     {
#       'name': 'image_path',
#       'type': 'image',
#       'encoder': {
#           'type': 'stacked_cnn',
#         }
#     }
#   ],
#   'output_features': [{'name': 'label', 'type': 'category'}],
#   'trainer': {'epochs': 5}
# }



# # Generates predictions and performance statistics for the test set.
# test_stats, predictions, output_directory = model.evaluate(
#   train_df,
#   collect_predictions=True,
#   collect_overall_stats=True
# )


# confusion_matrix(
#   [test_stats],
#   model.training_set_metadata,
#   'label',
#   top_n_classes=[5],
#   model_names=[''],
#   normalize=True,
# )

# # Visualizes learning curves, which show how performance metrics changed over
# # time during training.
# from ludwig.visualize import learning_curves

# learning_curves(train_stats, output_feature_name='label')

# predictions, output_directory = model.predict(train_df)