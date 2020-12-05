import pickle
import json
import pandas as pd

path = "/Users/weijiesun/cmput/692/692_project_webtable/source/tables.json"
# path = "test_data.json"
pd.set_option("display.max_rows", None, "display.max_columns", None)

tables = []
skip_num_row = 0
index = skip_num_row
with open(path, 'r') as f:
    for i in range(skip_num_row):
        try:
            a = f.readline()
        except:
            continue
    while True:
        try:
            a = f.readline()
            y = json.loads(a)
        except:
            index += 1
            continue
        columns_title = []
        for row in y['tableHeaders']:
            for value in row:
                columns_title.append(value['text'])

        data_list = []
        for rows in y['tableData']:
            row_list = []
            for row in rows:
                value = row['text']
                for link in row['surfaceLinks']:
                    value += ", " + link['target']['title'].replace('_', ' ')
                row_list.append(value)
            data_list.append(row_list)
        #df = pd.DataFrame(data_list, columns=columns)

        table_features = {}
        table_features['num_col'] = y['numCols']
        table_features['num_row'] = y['numDataRows']
        print('Table %d result' % index)
        if 'pgTitle' in y.keys():
            table_features['pg_title'] = y['pgTitle']
            print(y['pgTitle'])
        if 'sectionTitle' in y.keys():
            table_features['section_title'] = y['sectionTitle']
            print(y['sectionTitle'])
        if 'tableCaption' in y.keys():
            table_features['table_caption'] = y['tableCaption']
            print(y['tableCaption'])
        #
        print(columns_title)
        for row in data_list:
            print(row)
        label = input("Which class this table belongs to? (arts, society, games, sports, science, skip)")
        while label not in ['arts', 'society', 'sports', 'games', 'science', 'skip', 'end']:
            label = input("Which class this table belongs to? (arts, society, sports, games, science, skip)")
        if label == 'skip':
            print('skip this table')
            index += 1
            continue
        if label == 'end':
            print('Current index is %d' % index)
            break
        # table_features - dict, columns_title - list, data_list - list of lists, label - string
        tables.append([table_features, columns_title, data_list, label])
        index += 1
#data = []
with open('data/wiki_data_new.pickle', 'rb') as fp:
    data = pickle.load(fp)
    print(len(data))
print(len(tables))
with open('data/wiki_data_new.pickle', 'wb') as fp:
    data.extend(tables)
    print(len(data))
    pickle.dump(data, fp)