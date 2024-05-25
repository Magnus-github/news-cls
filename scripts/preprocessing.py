import pandas as pd
import json
import matplotlib.pyplot as plt


def inspect_data(data_folder: str = 'data/') -> None:
    train_data = pd.read_json(data_folder + 'train_prep.jsonl', lines=True)
    dev_data = pd.read_json(data_folder + 'dev_prep.jsonl', lines=True)
    test_data = pd.read_json(data_folder + 'test_prep.jsonl', lines=True)

    print('Train data:')
    print(train_data.head())
    print(train_data.columns)
    print('Dev data:')
    print(dev_data.head())
    print('Test data:')
    print(test_data.head())

    count_df = pd.DataFrame(train_data.groupby(["category"]).count()["headline"].sort_values(ascending=False),columns=["headline"])
    plt.figure(figsize=(50,20))
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.bar(count_df.index,count_df["headline"])
    plt.plot(count_df.index,count_df["headline"],'ro')
    plt.savefig('outputs/preprocessing/category_count.png')
    return None


def delete_empty(data_folder: str = 'data/') -> None:
    train_data = pd.read_json(data_folder + 'train.jsonl', lines=True)
    dev_data = pd.read_json(data_folder + 'dev.jsonl', lines=True)
    test_data = pd.read_json(data_folder + 'test.jsonl', lines=True)
    
    train_data = train_data.loc[train_data['headline'] != ""]
    train_data = train_data.loc[train_data['short_description'] != ""]
    dev_data = dev_data.loc[dev_data['headline'] != ""]
    dev_data = dev_data.loc[dev_data['short_description'] != ""]
    test_data = test_data.loc[test_data['headline'] != ""]
    test_data = test_data.loc[test_data['short_description'] != ""]
    
    train_data.to_json(data_folder + 'train_prep.jsonl', orient='records', lines=True)
    dev_data.to_json(data_folder + 'dev_prep.jsonl', orient='records', lines=True)
    test_data.to_json(data_folder + 'test_prep.jsonl', orient='records', lines=True)
    
    return None


def merge_data_cells(data_folder: str = 'data/') -> None:
    train_data = pd.read_json(data_folder + 'train_prep.jsonl', lines=True)
    dev_data = pd.read_json(data_folder + 'dev_prep.jsonl', lines=True)
    test_data = pd.read_json(data_folder + 'test_prep.jsonl', lines=True)
    
    train_data['full_text'] = train_data['headline'] + ' ' + train_data['short_description']
    dev_data['full_text'] = dev_data['headline'] + ' ' + dev_data['short_description']
    test_data['full_text'] = test_data['headline'] + ' ' + test_data['short_description']
    
    train_data.to_json(data_folder + 'train_prep.jsonl', orient='records', lines=True)
    dev_data.to_json(data_folder + 'dev_prep.jsonl', orient='records', lines=True)
    test_data.to_json(data_folder + 'test_prep.jsonl', orient='records', lines=True)
    
    return None


if __name__ == '__main__':
    inspect_data()
    # delete_empty(data_folder='data/')
    # merge_data_cells(data_folder='data/')
