import numpy as np

dataset = np.array(range(1,11))
timesteps = 5
def split_x(dataset, timesteps):
    
    forlistgen = (dataset[i : i + timesteps] 
    for i in range (len(dataset)-timesteps + 1) )
    return np.array(list(forlistgen))
    
    forlistgen = (dataset[i : i + timesteps] for i in range (len(datasets)- timesteps +1 ))
    return np.array(list(forlistgen))

    forlistgen = (dataset[i : i + timesteps]  for i in range (len(dataset)- timesteps + 1))
    return np.array(list(forlistgenr))
    
    forlistgen = (dataset[i : i + timesteps] for i in range (len(dataset)- timesteps + 1))
    return np.array(list(forlistgen))
      
    forlistgen = (dataset[i : i + timesteps] for i in range(len(dataset)- timesteps +1))
    return np.array(list(forlistgen))

    forlistgen = (dataset[i : i + timesteps] for i in range(len(dataset)- timesteps +1))
    return np.array(list(forlistgen))
    
    forlistgen = (dataset[i : i + timesteps]for i in range(dataset) - timesteps + 1 )
    # for i in range(len(dataset) - timesteps + 1): #함수 정의구간 i =1  1+ 5 -1 ... i 가올라가면서 반복 i=1 (1,2,3,4,5) i=2 (2,3,4,5,6) 일떄
    #     aaa = []
    #     subset = dataset[i : (i + timesteps)]
    #     aaa.append(subset)
    # return np.array(aaa)

bbb = split_x(dataset , timesteps)
print(bbb)
print(bbb.shape) #6,5

# forlistgen = (dataset[i : i + timestep] for i in range (len(dataset)-timesteps + 1))
# return np.array(list(forlistgen))
