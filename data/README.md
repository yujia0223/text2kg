### Datasets

These are the datasets we have used for training out models. Some, such as DocRED and FewRel contain the scripts used to push the dataset to HuggingFace, although these shouldn't need to be run as the datasets are already all uploaded to HuggingFace.

The prep_data files are used to prepare the data for training with autotrain. There are separate files for each prompt template we have used. Their usage is as follows:
1. Create a directory to store and organize the output data files, as all scripts only output train.csv and test.csv.
2. Navigate into the newly created directory.
3. Run the desired prep_data file and supply it with the following arguments:

```--data_path=PATH``` PATH may be a location on the machine or a HuggingFace dataset repository. This argument is mandatory.

```val_set_size=SIZE``` SIZE is any integer value, and defaults to zero.