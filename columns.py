import pandas as pd

data = {
    'age' : [20,23,48],
    'height' : [183,192,175],
    'weight' : [77,83,65]
}
indexName = ['슈퍼맨','스파이더맨','배트맨']

frame = pd.DataFrame(data, index = indexName)
frame